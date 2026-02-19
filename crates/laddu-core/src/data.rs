#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;
#[cfg(feature = "rayon")]
use accurate::{sum::Klein, traits::*};
use arrow::{
    array::{Float32Array, Float64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use auto_ops::impl_op_ex;
#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};
use oxyroot::{Branch, Named, ReaderTree, RootFile, WriterTree};
use parking_lot::Mutex;
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};
#[cfg(feature = "mpi")]
use parquet::file::metadata::ParquetMetaData;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    fs::File,
    ops::{Deref, DerefMut, Index, IndexMut},
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "mpi")]
type WorldHandle = SimpleCommunicator;
#[cfg(not(feature = "mpi"))]
type WorldHandle = ();

use crate::utils::get_bin_edges;
use crate::{
    utils::{
        variables::{IntoP4Selection, P4Selection, Variable, VariableExpression},
        vectors::Vec4,
    },
    LadduError, LadduResult,
};
use indexmap::{IndexMap, IndexSet};

/// An event that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitudes::Amplitude). This particular event contains the reaction
/// $`\gamma p \to K_S^0 K_S^0 p`$ with a polarized photon beam.
pub fn test_event() -> EventData {
    use crate::utils::vectors::*;
    let pol_magnitude = 0.38562805;
    let pol_angle = 0.05708078;
    EventData {
        p4s: vec![
            Vec3::new(0.0, 0.0, 8.747).with_mass(0.0),         // beam
            Vec3::new(0.119, 0.374, 0.222).with_mass(1.007),   // "proton"
            Vec3::new(-0.112, 0.293, 3.081).with_mass(0.498),  // "kaon"
            Vec3::new(-0.007, -0.667, 5.446).with_mass(0.498), // "kaon"
        ],
        aux: vec![pol_magnitude, pol_angle],
        weight: 0.48,
    }
}

/// Particle names used by [`test_dataset`].
pub const TEST_P4_NAMES: &[&str] = &["beam", "proton", "kshort1", "kshort2"];
/// Auxiliary scalar names used by [`test_dataset`].
pub const TEST_AUX_NAMES: &[&str] = &["pol_magnitude", "pol_angle"];

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

/// Raw event data in a [`Dataset`] containing all particle and auxiliary information.
///
/// An [`EventData`] instance owns the list of four-momenta (`p4s`), auxiliary scalars (`aux`),
/// and weight recorded for a particular collision event. Use [`Event`] when you need a
/// metadata-aware view with name-based helpers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventData {
    /// A list of four-momenta for each particle.
    pub p4s: Vec<Vec4>,
    /// A list of auxiliary scalar values associated with the event.
    pub aux: Vec<f64>,
    /// The weight given to the event.
    pub weight: f64,
}

impl Display for EventData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Event:")?;
        writeln!(f, "  p4s:")?;
        for p4 in &self.p4s {
            writeln!(f, "    {}", p4.to_p4_string())?;
        }
        writeln!(f, "  aux:")?;
        for (idx, value) in self.aux.iter().enumerate() {
            writeln!(f, "    aux[{idx}]: {value}")?;
        }
        writeln!(f, "  weight:")?;
        writeln!(f, "    {}", self.weight)?;
        Ok(())
    }
}

impl EventData {
    /// Return a four-momentum from the sum of four-momenta at the given indices in the [`EventData`].
    pub fn get_p4_sum<T: AsRef<[usize]>>(&self, indices: T) -> Vec4 {
        indices.as_ref().iter().map(|i| self.p4s[*i]).sum::<Vec4>()
    }
    /// Boost all the four-momenta in the [`EventData`] to the rest frame of the given set of
    /// four-momenta by indices.
    pub fn boost_to_rest_frame_of<T: AsRef<[usize]>>(&self, indices: T) -> Self {
        let frame = self.get_p4_sum(indices);
        EventData {
            p4s: self
                .p4s
                .iter()
                .map(|p4| p4.boost(&(-frame.beta())))
                .collect(),
            aux: self.aux.clone(),
            weight: self.weight,
        }
    }
    /// Evaluate a [`Variable`] on an [`EventData`].
    pub fn evaluate<V: Variable>(&self, variable: &V) -> f64 {
        variable.value(self)
    }
}

/// Read-only event access interface for storage-agnostic compute paths.
///
/// This supports both row-backed (`EventData`) and columnar-backed row views.
pub trait EventAccess {
    /// Get four-momentum by positional index.
    fn p4_at(&self, p4_index: usize) -> Option<Vec4>;
    /// Get auxiliary scalar by positional index.
    fn aux_at(&self, aux_index: usize) -> Option<f64>;
    /// Get event weight.
    fn weight(&self) -> f64;
    /// Number of four-momentum entries available in this event.
    fn n_p4(&self) -> usize;
    /// Number of auxiliary scalar entries available in this event.
    fn n_aux(&self) -> usize;
    /// Sum selected four-momenta by positional indices.
    fn get_p4_sum<T: AsRef<[usize]>>(&self, indices: T) -> Option<Vec4>
    where
        Self: Sized,
    {
        indices
            .as_ref()
            .iter()
            .map(|index| self.p4_at(*index))
            .collect::<Option<Vec<_>>>()
            .map(|momenta| momenta.into_iter().sum())
    }
}

/// Read-only named event access for metadata-aware compute paths.
pub trait NamedEventAccess: EventAccess {
    /// Get four-momentum (or alias sum) by registered name.
    fn p4_by_name(&self, name: &str) -> Option<Vec4>;
    /// Get auxiliary scalar by registered name.
    fn aux_by_name(&self, name: &str) -> Option<f64>;
}

impl EventAccess for EventData {
    fn p4_at(&self, p4_index: usize) -> Option<Vec4> {
        self.p4s.get(p4_index).copied()
    }

    fn aux_at(&self, aux_index: usize) -> Option<f64> {
        self.aux.get(aux_index).copied()
    }

    fn weight(&self) -> f64 {
        self.weight
    }

    fn n_p4(&self) -> usize {
        self.p4s.len()
    }

    fn n_aux(&self) -> usize {
        self.aux.len()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct ColumnarP4Column {
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    e: Vec<f64>,
}

#[allow(dead_code)]
impl ColumnarP4Column {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            px: Vec::with_capacity(capacity),
            py: Vec::with_capacity(capacity),
            pz: Vec::with_capacity(capacity),
            e: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, p4: Vec4) {
        self.px.push(p4.x);
        self.py.push(p4.y);
        self.pz.push(p4.z);
        self.e.push(p4.t);
    }

    fn get(&self, event_index: usize) -> Vec4 {
        Vec4::new(
            self.px[event_index],
            self.py[event_index],
            self.pz[event_index],
            self.e[event_index],
        )
    }
}

/// Columnar dataset storage used by columnar execution paths.
///
/// O081 migration note: this storage model is the canonical target for the public
/// [`Dataset`] API. Row-backed storage remains transitional during migration.
#[derive(Debug, Default)]
pub(crate) struct DatasetStorage {
    metadata: Arc<DatasetMetadata>,
    p4: Vec<ColumnarP4Column>,
    aux: Vec<Vec<f64>>,
    weights: Vec<f64>,
}

impl Clone for DatasetStorage {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            p4: self.p4.clone(),
            aux: self.aux.clone(),
            weights: self.weights.clone(),
        }
    }
}

impl DatasetStorage {
    /// Build a columnar dataset from an event-row dataset.
    pub(crate) fn from_dataset(dataset: &Dataset) -> LadduResult<Self> {
        let n_events = dataset.events.len();
        let (n_p4, n_aux) = match dataset.events.first() {
            Some(first) => (first.p4s.len(), first.aux.len()),
            None => (
                dataset.metadata.p4_names.len(),
                dataset.metadata.aux_names.len(),
            ),
        };
        let mut p4 = (0..n_p4)
            .map(|_| ColumnarP4Column::with_capacity(n_events))
            .collect::<Vec<_>>();
        let mut aux = (0..n_aux)
            .map(|_| Vec::with_capacity(n_events))
            .collect::<Vec<_>>();
        let mut weights = Vec::with_capacity(n_events);
        for (event_index, event) in dataset.events.iter().enumerate() {
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
        Ok(Self {
            metadata: dataset.metadata.clone(),
            p4,
            aux,
            weights,
        })
    }

    /// Convert this columnar dataset back to a row-event dataset.
    pub(crate) fn to_dataset(&self) -> Dataset {
        let events = (0..self.n_events())
            .map(|event_index| Arc::new(self.event_data(event_index)))
            .collect::<Vec<_>>();
        Dataset::new_local(events, self.metadata.clone())
    }

    /// Access metadata.
    pub(crate) fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Number of local events.
    pub(crate) fn n_events(&self) -> usize {
        self.weights.len()
    }

    /// Retrieve a p4 value by row and p4 index.
    ///
    /// Panics when either index is out of bounds.
    pub(crate) fn p4(&self, event_index: usize, p4_index: usize) -> Vec4 {
        self.p4[p4_index].get(event_index)
    }

    /// Retrieve an aux value by row and aux index.
    ///
    /// Panics when either index is out of bounds.
    pub(crate) fn aux(&self, event_index: usize, aux_index: usize) -> f64 {
        self.aux[aux_index][event_index]
    }

    /// Retrieve event weight by row index.
    ///
    /// Panics when the row index is out of bounds.
    pub(crate) fn weight(&self, event_index: usize) -> f64 {
        self.weights[event_index]
    }

    pub(crate) fn event_data(&self, event_index: usize) -> EventData {
        let mut p4s = Vec::with_capacity(self.p4.len());
        for p4_index in 0..self.p4.len() {
            p4s.push(self.p4(event_index, p4_index));
        }
        let mut aux = Vec::with_capacity(self.aux.len());
        for aux_index in 0..self.aux.len() {
            aux.push(self.aux(event_index, aux_index));
        }
        EventData {
            p4s,
            aux,
            weight: self.weight(event_index),
        }
    }

    fn row_view(&self, event_index: usize) -> ColumnarEventView<'_> {
        ColumnarEventView {
            storage: self,
            event_index,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn for_each_named_event_local<F>(&self, mut op: F)
    where
        F: FnMut(usize, NamedEventView<'_>),
    {
        for event_index in 0..self.n_events() {
            let row = self.row_view(event_index);
            let view = NamedEventView {
                row,
                metadata: &self.metadata,
            };
            op(event_index, view);
        }
    }

    pub(crate) fn event_view(&self, event_index: usize) -> NamedEventView<'_> {
        let row = self.row_view(event_index);
        NamedEventView {
            row,
            metadata: self.metadata(),
        }
    }

    fn write_parquet_impl(
        &self,
        file_path: PathBuf,
        options: &DatasetWriteOptions,
    ) -> LadduResult<()> {
        let batch_size = options.batch_size.max(1);
        let precision = options.precision;
        let schema = Arc::new(build_parquet_schema(&self.metadata, precision));

        #[cfg(feature = "mpi")]
        let is_root = crate::mpi::get_world()
            .as_ref()
            .map_or(true, |world| world.rank() == 0);
        #[cfg(not(feature = "mpi"))]
        let is_root = true;

        let mut writer: Option<ArrowWriter<File>> = None;
        if is_root {
            let file = File::create(&file_path)?;
            writer = Some(
                ArrowWriter::try_new(file, schema.clone(), None).map_err(|err| {
                    LadduError::Custom(format!("Failed to create Parquet writer: {err}"))
                })?,
            );
        }

        let n_rows = self.n_events();
        let mut start = 0usize;
        while start < n_rows {
            let end = (start + batch_size).min(n_rows);
            if let Some(writer) = writer.as_mut() {
                let batch =
                    columnar_range_to_record_batch(self, start, end, schema.clone(), precision)
                        .map_err(|err| {
                            LadduError::Custom(format!("Failed to build Parquet batch: {err}"))
                        })?;
                writer.write(&batch).map_err(|err| {
                    LadduError::Custom(format!("Failed to write Parquet batch: {err}"))
                })?;
            }
            start = end;
        }

        if let Some(writer) = writer {
            writer.close().map_err(|err| {
                LadduError::Custom(format!("Failed to finalise Parquet file: {err}"))
            })?;
        }

        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Debug)]
struct ColumnarEventView<'a> {
    storage: &'a DatasetStorage,
    event_index: usize,
}

#[allow(dead_code)]
impl ColumnarEventView<'_> {
    fn p4(&self, p4_index: usize) -> Vec4 {
        self.storage.p4(self.event_index, p4_index)
    }

    fn aux(&self, aux_index: usize) -> f64 {
        self.storage.aux(self.event_index, aux_index)
    }

    fn weight(&self) -> f64 {
        self.storage.weight(self.event_index)
    }

    fn get_p4_sum<T: AsRef<[usize]>>(&self, indices: T) -> Vec4 {
        indices.as_ref().iter().map(|index| self.p4(*index)).sum()
    }
}

impl EventAccess for ColumnarEventView<'_> {
    fn p4_at(&self, p4_index: usize) -> Option<Vec4> {
        Some(self.p4(p4_index))
    }

    fn aux_at(&self, aux_index: usize) -> Option<f64> {
        Some(self.aux(aux_index))
    }

    fn weight(&self) -> f64 {
        ColumnarEventView::weight(self)
    }

    fn n_p4(&self) -> usize {
        self.storage.p4.len()
    }

    fn n_aux(&self) -> usize {
        self.storage.aux.len()
    }
}

/// A name-aware columnar event view over a single row in a dataset.
#[derive(Debug)]
pub struct NamedEventView<'a> {
    row: ColumnarEventView<'a>,
    metadata: &'a DatasetMetadata,
}

impl NamedEventView<'_> {
    /// Retrieve a four-momentum by metadata name.
    pub fn p4(&self, name: &str) -> Option<Vec4> {
        let selection = self.metadata.p4_selection(name)?;
        Some(
            selection
                .indices()
                .iter()
                .map(|index| self.row.p4(*index))
                .sum(),
        )
    }

    /// Retrieve an auxiliary scalar by metadata name.
    pub fn aux(&self, name: &str) -> Option<f64> {
        let index = self.metadata.aux_index(name)?;
        Some(self.row.aux(index))
    }

    /// Retrieve event weight.
    pub fn weight(&self) -> f64 {
        self.row.weight()
    }

    /// Retrieve the sum of multiple four-momenta selected by name.
    pub fn get_p4_sum<N>(&self, names: N) -> Option<Vec4>
    where
        N: IntoIterator,
        N::Item: AsRef<str>,
    {
        names
            .into_iter()
            .map(|name| self.p4(name.as_ref()))
            .collect::<Option<Vec<_>>>()
            .map(|momenta| momenta.into_iter().sum())
    }

    /// Evaluate a [`Variable`] against this event.
    pub fn evaluate<V: Variable>(&self, variable: &V) -> f64 {
        variable.value_view(self)
    }
}

impl EventAccess for NamedEventView<'_> {
    fn p4_at(&self, p4_index: usize) -> Option<Vec4> {
        Some(self.row.p4(p4_index))
    }

    fn aux_at(&self, aux_index: usize) -> Option<f64> {
        Some(self.row.aux(aux_index))
    }

    fn weight(&self) -> f64 {
        self.row.weight()
    }

    fn n_p4(&self) -> usize {
        self.row.storage.p4.len()
    }

    fn n_aux(&self) -> usize {
        self.row.storage.aux.len()
    }
}

impl NamedEventAccess for NamedEventView<'_> {
    fn p4_by_name(&self, name: &str) -> Option<Vec4> {
        self.p4(name)
    }

    fn aux_by_name(&self, name: &str) -> Option<f64> {
        self.aux(name)
    }
}

/// A collection of [`EventData`].
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub(crate) p4_names: Vec<String>,
    pub(crate) aux_names: Vec<String>,
    pub(crate) p4_lookup: IndexMap<String, usize>,
    pub(crate) aux_lookup: IndexMap<String, usize>,
    pub(crate) p4_selections: IndexMap<String, P4Selection>,
}

impl DatasetMetadata {
    /// Construct metadata from explicit particle and auxiliary names.
    pub fn new<P: Into<String>, A: Into<String>>(
        p4_names: Vec<P>,
        aux_names: Vec<A>,
    ) -> LadduResult<Self> {
        let mut p4_lookup = IndexMap::with_capacity(p4_names.len());
        let mut aux_lookup = IndexMap::with_capacity(aux_names.len());
        let mut p4_selections = IndexMap::with_capacity(p4_names.len());
        let p4_names: Vec<String> = p4_names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| {
                let name = name.into();
                if p4_lookup.contains_key(&name) {
                    return Err(LadduError::DuplicateName {
                        category: "p4",
                        name,
                    });
                }
                p4_lookup.insert(name.clone(), idx);
                p4_selections.insert(
                    name.clone(),
                    P4Selection::with_indices(vec![name.clone()], vec![idx]),
                );
                Ok(name)
            })
            .collect::<Result<_, _>>()?;
        let aux_names: Vec<String> = aux_names
            .into_iter()
            .enumerate()
            .map(|(idx, name)| {
                let name = name.into();
                if aux_lookup.contains_key(&name) {
                    return Err(LadduError::DuplicateName {
                        category: "aux",
                        name,
                    });
                }
                aux_lookup.insert(name.clone(), idx);
                Ok(name)
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            p4_names,
            aux_names,
            p4_lookup,
            aux_lookup,
            p4_selections,
        })
    }

    /// Create metadata with no registered names.
    pub fn empty() -> Self {
        Self {
            p4_names: Vec::new(),
            aux_names: Vec::new(),
            p4_lookup: IndexMap::new(),
            aux_lookup: IndexMap::new(),
            p4_selections: IndexMap::new(),
        }
    }

    /// Resolve the index of a four-momentum by name.
    pub fn p4_index(&self, name: &str) -> Option<usize> {
        self.p4_lookup.get(name).copied()
    }

    /// Registered four-momentum names in declaration order.
    pub fn p4_names(&self) -> &[String] {
        &self.p4_names
    }

    /// Resolve the index of an auxiliary scalar by name.
    pub fn aux_index(&self, name: &str) -> Option<usize> {
        self.aux_lookup.get(name).copied()
    }

    /// Registered auxiliary scalar names in declaration order.
    pub fn aux_names(&self) -> &[String] {
        &self.aux_names
    }

    /// Look up a resolved four-momentum selection by name (canonical or alias).
    pub fn p4_selection(&self, name: &str) -> Option<&P4Selection> {
        self.p4_selections.get(name)
    }

    /// Register an alias mapping to one or more existing four-momenta.
    pub fn add_p4_alias<N>(&mut self, alias: N, mut selection: P4Selection) -> LadduResult<()>
    where
        N: Into<String>,
    {
        let alias = alias.into();
        if self.p4_selections.contains_key(&alias) {
            return Err(LadduError::DuplicateName {
                category: "alias",
                name: alias,
            });
        }
        selection.bind(self)?;
        self.p4_selections.insert(alias, selection);
        Ok(())
    }

    /// Register multiple aliases at once.
    pub fn add_p4_aliases<I, N>(&mut self, entries: I) -> LadduResult<()>
    where
        I: IntoIterator<Item = (N, P4Selection)>,
        N: Into<String>,
    {
        for (alias, selection) in entries {
            self.add_p4_alias(alias, selection)?;
        }
        Ok(())
    }

    pub(crate) fn append_indices_for_name(
        &self,
        name: &str,
        target: &mut Vec<usize>,
    ) -> LadduResult<()> {
        if let Some(selection) = self.p4_selections.get(name) {
            target.extend_from_slice(selection.indices());
            return Ok(());
        }
        Err(LadduError::UnknownName {
            category: "p4",
            name: name.to_string(),
        })
    }
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self::empty()
    }
}

/// A collection of events with optional metadata for name-based lookups.
///
/// O081 migration note: the public `Dataset` name is planned to become columnar-backed.
/// The current row-backed representation is transitional until full adoption is complete.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// The [`EventData`] contained in the [`Dataset`]
    events: Vec<Event>,
    pub(crate) columnar: DatasetStorage,
    pub(crate) metadata: Arc<DatasetMetadata>,
}

/// Metadata-aware view of an [`EventData`] with name-based helpers.
#[derive(Clone, Debug)]
pub struct Event {
    event: Arc<EventData>,
    metadata: Arc<DatasetMetadata>,
}

impl Event {
    /// Create a new metadata-aware event from raw data and dataset metadata.
    pub fn new(event: Arc<EventData>, metadata: Arc<DatasetMetadata>) -> Self {
        Self { event, metadata }
    }

    /// Borrow the raw [`EventData`].
    pub fn data(&self) -> &EventData {
        &self.event
    }

    /// Obtain a clone of the underlying [`EventData`] handle.
    pub fn data_arc(&self) -> Arc<EventData> {
        self.event.clone()
    }

    /// Return the four-momenta stored in this event keyed by their registered names.
    pub fn p4s(&self) -> IndexMap<&str, Vec4> {
        let mut map = IndexMap::with_capacity(self.metadata.p4_names.len());
        for (idx, name) in self.metadata.p4_names.iter().enumerate() {
            if let Some(p4) = self.event.p4s.get(idx) {
                map.insert(name.as_str(), *p4);
            }
        }
        map
    }

    /// Return the auxiliary scalars stored in this event keyed by their registered names.
    pub fn aux(&self) -> IndexMap<&str, f64> {
        let mut map = IndexMap::with_capacity(self.metadata.aux_names.len());
        for (idx, name) in self.metadata.aux_names.iter().enumerate() {
            if let Some(value) = self.event.aux.get(idx) {
                map.insert(name.as_str(), *value);
            }
        }
        map
    }

    /// Return the event weight.
    pub fn weight(&self) -> f64 {
        self.event.weight
    }

    /// Retrieve the dataset metadata attached to this event.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Clone the metadata handle associated with this event.
    pub fn metadata_arc(&self) -> Arc<DatasetMetadata> {
        self.metadata.clone()
    }

    /// Retrieve a four-momentum (or aliased sum) by name.
    pub fn p4(&self, name: &str) -> Option<Vec4> {
        self.metadata
            .p4_selection(name)
            .map(|selection| selection.momentum(&self.event))
    }

    fn resolve_p4_indices<N>(&self, names: N) -> Vec<usize>
    where
        N: IntoIterator,
        N::Item: AsRef<str>,
    {
        let mut indices = Vec::new();
        for name in names {
            let name_ref = name.as_ref();
            if let Some(selection) = self.metadata.p4_selection(name_ref) {
                indices.extend_from_slice(selection.indices());
            } else {
                panic!("Unknown particle name '{name}'", name = name_ref);
            }
        }
        indices
    }

    /// Return a four-momentum formed by summing four-momenta with the specified names.
    pub fn get_p4_sum<N>(&self, names: N) -> Vec4
    where
        N: IntoIterator,
        N::Item: AsRef<str>,
    {
        let indices = self.resolve_p4_indices(names);
        self.event.get_p4_sum(&indices)
    }

    /// Boost all four-momenta into the rest frame defined by the specified particle names.
    pub fn boost_to_rest_frame_of<N>(&self, names: N) -> EventData
    where
        N: IntoIterator,
        N::Item: AsRef<str>,
    {
        let indices = self.resolve_p4_indices(names);
        self.event.boost_to_rest_frame_of(&indices)
    }

    /// Evaluate a [`Variable`] over this event.
    pub fn evaluate<V: Variable>(&self, variable: &V) -> f64 {
        self.event.evaluate(variable)
    }
}

impl EventAccess for Event {
    fn p4_at(&self, p4_index: usize) -> Option<Vec4> {
        self.event.p4s.get(p4_index).copied()
    }

    fn aux_at(&self, aux_index: usize) -> Option<f64> {
        self.event.aux.get(aux_index).copied()
    }

    fn weight(&self) -> f64 {
        self.event.weight
    }

    fn n_p4(&self) -> usize {
        self.event.p4s.len()
    }

    fn n_aux(&self) -> usize {
        self.event.aux.len()
    }
}

impl NamedEventAccess for Event {
    fn p4_by_name(&self, name: &str) -> Option<Vec4> {
        self.p4(name)
    }

    fn aux_by_name(&self, name: &str) -> Option<f64> {
        self.metadata
            .aux_index(name)
            .and_then(|index| self.event.aux.get(index).copied())
    }
}

impl Deref for Event {
    type Target = EventData;

    fn deref(&self) -> &Self::Target {
        &self.event
    }
}

impl AsRef<EventData> for Event {
    fn as_ref(&self) -> &EventData {
        self.data()
    }
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
                });
            }
        }
        DatasetIntoIter::Local(self.events.into_iter())
    }
}

impl Dataset {
    /// Borrow locally stored events.
    pub fn events_local(&self) -> &[Event] {
        &self.events
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
                return DatasetIter::Mpi(DatasetMpiIter {
                    dataset: self,
                    world,
                    index: 0,
                    total: self.n_events(),
                });
            }
        }
        DatasetIter::Local(self.events.iter())
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

    /// Retrieve a single event by index.
    pub fn event(&self, index: usize) -> LadduResult<Event> {
        self.get_event(index).ok_or_else(|| {
            LadduError::Custom(format!(
                "Dataset index out of bounds: index {index}, length {}",
                self.n_events()
            ))
        })
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
                events[range.clone()].iter().cloned().collect()
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

#[cfg(feature = "mpi")]
/// Iterator over a [`Dataset`] that fetches events across MPI ranks.
pub struct DatasetMpiIter<'a> {
    dataset: &'a Dataset,
    world: SimpleCommunicator,
    index: usize,
    total: usize,
}

#[cfg(feature = "mpi")]
impl<'a> Iterator for DatasetMpiIter<'a> {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let event = fetch_event_mpi(self.dataset, self.index, &self.world, self.total);
        self.index += 1;
        Some(event)
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
}

#[cfg(feature = "mpi")]
impl Iterator for DatasetMpiIntoIter {
    type Item = Event;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }
        let event = fetch_event_mpi_from_events(
            &self.events,
            &self.metadata,
            self.index,
            &self.world,
            self.total,
        );
        self.index += 1;
        Some(event)
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
fn fetch_event_mpi_from_events(
    events: &[Event],
    metadata: &Arc<DatasetMetadata>,
    global_index: usize,
    world: &SimpleCommunicator,
    total: usize,
) -> Event {
    fetch_event_mpi_generic(global_index, total, world, metadata, |local_index| {
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
    let config = bincode::config::standard();
    if world.rank() == owning_rank {
        let event = local_event(local_index);
        serialized_event_buffer = bincode::serde::encode_to_vec(event.data(), config).unwrap();
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
        let (event, _): (EventData, usize) =
            bincode::serde::decode_from_slice(&serialized_event_buffer[..], config).unwrap();
        Event::new(Arc::new(event), metadata.clone())
    }
}

impl Dataset {
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
        let columnar = Self::columnar_from_wrapped_events(&wrapped_events, metadata.clone())
            .expect("Dataset requires rectangular p4/aux columns for canonical columnar storage");
        Dataset {
            events: wrapped_events,
            columnar,
            metadata,
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
        Dataset {
            events: local,
            columnar,
            metadata,
        }
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

    /// Convert this row-event dataset to columnar layout.
    pub(crate) fn to_storage(&self) -> LadduResult<DatasetStorage> {
        Ok(self.columnar.clone())
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
    pub fn n_events_mpi(&self, world: &SimpleCommunicator) -> usize {
        let mut n_events_partitioned: Vec<usize> = vec![0; world.size() as usize];
        let n_events_local = self.n_events_local();
        world.all_gather_into(&n_events_local, &mut n_events_partitioned);
        n_events_partitioned.iter().sum()
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

    /// Returns the sum of the weights for each [`EventData`] in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events_weighted`] instead.
    pub fn n_events_weighted_local(&self) -> f64 {
        #[cfg(feature = "rayon")]
        return self
            .columnar
            .weights
            .par_iter()
            .copied()
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        return self.columnar.weights.iter().sum();
    }
    /// Returns the sum of the weights for each [`EventData`] in the [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events_weighted`] instead.
    #[cfg(feature = "mpi")]
    pub fn n_events_weighted_mpi(&self, world: &SimpleCommunicator) -> f64 {
        let mut n_events_weighted_partitioned: Vec<f64> = vec![0.0; world.size() as usize];
        let n_events_weighted_local = self.n_events_weighted_local();
        world.all_gather_into(&n_events_weighted_local, &mut n_events_weighted_partitioned);
        #[cfg(feature = "rayon")]
        return n_events_weighted_partitioned
            .into_par_iter()
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        return n_events_weighted_partitioned.iter().sum();
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
        let filtered_events: Vec<Arc<EventData>> = self
            .events
            .par_iter()
            .filter(|event| compiled.evaluate(event.as_ref()))
            .map(|event| event.data_arc())
            .collect();
        #[cfg(not(feature = "rayon"))]
        let filtered_events: Vec<Arc<EventData>> = self
            .events
            .iter()
            .filter(|event| compiled.evaluate(event.as_ref()))
            .map(|event| event.data_arc())
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
        let evaluated: Vec<(usize, Arc<EventData>)> = self
            .events
            .par_iter()
            .filter_map(|event| {
                let value = variable.value(event.as_ref());
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, event.data_arc()))
                } else {
                    None
                }
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let evaluated: Vec<(usize, Arc<EventData>)> = self
            .events
            .iter()
            .filter_map(|event| {
                let value = variable.value(event.as_ref());
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, event.data_arc()))
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

    fn write_parquet_impl(
        &self,
        file_path: PathBuf,
        options: &DatasetWriteOptions,
    ) -> LadduResult<()> {
        self.columnar.write_parquet_impl(file_path, options)
    }

    fn write_root_impl(
        &self,
        file_path: PathBuf,
        options: &DatasetWriteOptions,
    ) -> LadduResult<()> {
        let tree_name = options.tree.clone().unwrap_or_else(|| "events".to_string());
        let branch_count = self.metadata.p4_names.len() * 4 + self.metadata.aux_names.len() + 1; // +weight

        #[cfg(feature = "mpi")]
        let mut world_opt = crate::mpi::get_world();
        #[cfg(feature = "mpi")]
        let is_root = world_opt.as_ref().map_or(true, |world| world.rank() == 0);
        #[cfg(not(feature = "mpi"))]
        let is_root = true;

        #[cfg(feature = "mpi")]
        let world: Option<WorldHandle> = world_opt.take();
        #[cfg(not(feature = "mpi"))]
        let world: Option<WorldHandle> = None;

        let total_events = self.n_events();
        let dataset_arc = Arc::new(self.clone());

        match options.precision {
            FloatPrecision::F64 => self.write_root_with_type::<f64>(
                dataset_arc,
                world,
                is_root,
                &file_path,
                &tree_name,
                branch_count,
                total_events,
            ),
            FloatPrecision::F32 => self.write_root_with_type::<f32>(
                dataset_arc,
                world,
                is_root,
                &file_path,
                &tree_name,
                branch_count,
                total_events,
            ),
        }
    }
}

fn canonicalize_dataset_path(file_path: &str) -> LadduResult<PathBuf> {
    Ok(Path::new(&*shellexpand::full(file_path)?).canonicalize()?)
}

fn expand_output_path(file_path: &str) -> LadduResult<PathBuf> {
    Ok(PathBuf::from(&*shellexpand::full(file_path)?))
}

/// Load a [`Dataset`] from a Parquet file.
pub fn read_parquet(file_path: &str, options: &DatasetReadOptions) -> LadduResult<Arc<Dataset>> {
    let storage = read_parquet_storage(file_path, options)?;
    Ok(Arc::new(storage.to_dataset()))
}

/// Load a [`DatasetStorage`] from a Parquet file without materializing row-event storage.
pub(crate) fn read_parquet_storage(
    file_path: &str,
    options: &DatasetReadOptions,
) -> LadduResult<Arc<DatasetStorage>> {
    let path = canonicalize_dataset_path(file_path)?;
    let (detected_p4_names, detected_aux_names) = detect_columns(&path)?;
    let metadata = options.resolve_metadata(detected_p4_names, detected_aux_names)?;
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

    #[cfg(feature = "mpi")]
    {
        if let Some(world) = crate::mpi::get_world() {
            return read_parquet_columnar_mpi(builder, metadata, &world);
        }
    }

    read_parquet_columnar_local(builder, metadata)
}

fn read_parquet_local(
    builder: ParquetRecordBatchReaderBuilder<File>,
    metadata: Arc<DatasetMetadata>,
) -> LadduResult<Arc<Dataset>> {
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
    let events = batches_to_events(batches, metadata.as_ref())?;
    Ok(Arc::new(Dataset::new_with_metadata(events, metadata)))
}

fn read_parquet_columnar_local(
    builder: ParquetRecordBatchReaderBuilder<File>,
    metadata: Arc<DatasetMetadata>,
) -> LadduResult<Arc<DatasetStorage>> {
    let total_rows = builder.metadata().file_metadata().num_rows() as usize;
    if total_rows == 0 {
        return Ok(Arc::new(empty_dataset_columnar(metadata)));
    }

    let reader = builder.build()?;
    let mut p4 = (0..metadata.p4_names.len())
        .map(|_| ColumnarP4Column::with_capacity(total_rows))
        .collect::<Vec<_>>();
    let mut aux = (0..metadata.aux_names.len())
        .map(|_| Vec::with_capacity(total_rows))
        .collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(total_rows);

    for batch in reader {
        let batch = batch?;
        append_record_batch_to_columnar(
            &batch,
            metadata.as_ref(),
            &mut p4,
            &mut aux,
            &mut weights,
        )?;
    }

    Ok(Arc::new(DatasetStorage {
        metadata,
        p4,
        aux,
        weights,
    }))
}

#[cfg(feature = "mpi")]
fn read_parquet_mpi(
    mut builder: ParquetRecordBatchReaderBuilder<File>,
    metadata: Arc<DatasetMetadata>,
    world: &SimpleCommunicator,
) -> LadduResult<Arc<Dataset>> {
    let parquet_metadata = builder.metadata().clone();
    let total_rows = parquet_metadata.file_metadata().num_rows() as usize;
    if total_rows == 0 {
        return Ok(Arc::new(Dataset::new_local(Vec::new(), metadata)));
    }

    let partition = world.partition(total_rows);
    let rank = world.rank() as usize;
    let local_range = partition.range_for_rank(rank);
    let local_start = local_range.start;
    let local_end = local_range.end;
    if local_start == local_end {
        return Ok(Arc::new(Dataset::new_local(Vec::new(), metadata)));
    }

    let (row_groups, first_row_start) =
        row_groups_for_range(&parquet_metadata, local_start, local_end);
    if !row_groups.is_empty() {
        builder = builder.with_row_groups(row_groups);
    }

    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;
    let mut events = batches_to_events(batches, metadata.as_ref())?;

    let drop_front = local_start.saturating_sub(first_row_start);
    if drop_front > 0 {
        events.drain(0..drop_front);
    }
    let expected_local = local_end - local_start;
    if events.len() > expected_local {
        events.truncate(expected_local);
    }
    if events.len() != expected_local {
        return Err(LadduError::LengthMismatch {
            context: format!("Loaded rows for MPI rank {rank}"),
            expected: expected_local,
            actual: events.len(),
        });
    }

    Ok(Arc::new(Dataset::new_local(events, metadata)))
}

#[cfg(feature = "mpi")]
fn read_parquet_columnar_mpi(
    mut builder: ParquetRecordBatchReaderBuilder<File>,
    metadata: Arc<DatasetMetadata>,
    world: &SimpleCommunicator,
) -> LadduResult<Arc<DatasetStorage>> {
    let parquet_metadata = builder.metadata().clone();
    let total_rows = parquet_metadata.file_metadata().num_rows() as usize;
    if total_rows == 0 {
        return Ok(Arc::new(empty_dataset_columnar(metadata)));
    }

    let partition = world.partition(total_rows);
    let rank = world.rank() as usize;
    let local_range = partition.range_for_rank(rank);
    let local_start = local_range.start;
    let local_end = local_range.end;
    if local_start == local_end {
        return Ok(Arc::new(empty_dataset_columnar(metadata)));
    }

    let (row_groups, first_row_start) =
        row_groups_for_range(&parquet_metadata, local_start, local_end);
    if !row_groups.is_empty() {
        builder = builder.with_row_groups(row_groups);
    }

    let reader = builder.build()?;
    let mut p4 = (0..metadata.p4_names.len())
        .map(|_| ColumnarP4Column::with_capacity(local_end - first_row_start))
        .collect::<Vec<_>>();
    let mut aux = (0..metadata.aux_names.len())
        .map(|_| Vec::with_capacity(local_end - first_row_start))
        .collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(local_end - first_row_start);
    for batch in reader {
        let batch = batch?;
        append_record_batch_to_columnar(
            &batch,
            metadata.as_ref(),
            &mut p4,
            &mut aux,
            &mut weights,
        )?;
    }
    let mut columnar = DatasetStorage {
        metadata,
        p4,
        aux,
        weights,
    };

    let drop_front = local_start.saturating_sub(first_row_start);
    let expected_local = local_end - local_start;
    trim_columnar_rows(&mut columnar, drop_front, expected_local);
    if columnar.n_events() != expected_local {
        return Err(LadduError::LengthMismatch {
            context: format!("Loaded rows for MPI rank {rank}"),
            expected: expected_local,
            actual: columnar.n_events(),
        });
    }

    Ok(Arc::new(columnar))
}

#[cfg(feature = "mpi")]
fn row_groups_for_range(
    metadata: &Arc<ParquetMetaData>,
    start: usize,
    end: usize,
) -> (Vec<usize>, usize) {
    let mut selected = Vec::new();
    let mut first_row_start = start;
    let mut offset = 0usize;

    for (idx, row_group) in metadata.row_groups().iter().enumerate() {
        let group_start = offset;
        let rows = row_group.num_rows() as usize;
        let group_end = group_start + rows;
        offset = group_end;

        if group_end <= start {
            continue;
        }
        if group_start >= end {
            break;
        }
        if selected.is_empty() {
            first_row_start = group_start;
        }
        selected.push(idx);
        if group_end >= end {
            break;
        }
    }

    (selected, first_row_start)
}

fn batches_to_events(
    batches: Vec<RecordBatch>,
    metadata: &DatasetMetadata,
) -> LadduResult<Vec<Arc<EventData>>> {
    #[cfg(feature = "rayon")]
    {
        let batch_events: Vec<LadduResult<Vec<Arc<EventData>>>> = batches
            .into_par_iter()
            .map(|batch| record_batch_to_events(batch, &metadata.p4_names, &metadata.aux_names))
            .collect();
        let mut events = Vec::new();
        for batch in batch_events {
            let mut batch = batch?;
            events.append(&mut batch);
        }
        Ok(events)
    }

    #[cfg(not(feature = "rayon"))]
    {
        Ok(batches
            .into_iter()
            .map(|batch| record_batch_to_events(batch, &metadata.p4_names, &metadata.aux_names))
            .collect::<LadduResult<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect())
    }
}

fn empty_dataset_columnar(metadata: Arc<DatasetMetadata>) -> DatasetStorage {
    DatasetStorage {
        p4: (0..metadata.p4_names.len())
            .map(|_| ColumnarP4Column::with_capacity(0))
            .collect(),
        aux: (0..metadata.aux_names.len())
            .map(|_| Vec::with_capacity(0))
            .collect(),
        weights: Vec::new(),
        metadata,
    }
}

fn append_record_batch_to_columnar(
    batch: &RecordBatch,
    metadata: &DatasetMetadata,
    p4_columns_out: &mut [ColumnarP4Column],
    aux_columns_out: &mut [Vec<f64>],
    weights_out: &mut Vec<f64>,
) -> LadduResult<()> {
    let p4_columns: Vec<P4Columns<'_>> = metadata
        .p4_names
        .iter()
        .map(|name| prepare_p4_columns(batch, name))
        .collect::<Result<_, _>>()?;
    let aux_columns: Vec<FloatColumn<'_>> = metadata
        .aux_names
        .iter()
        .map(|name| prepare_float_column(batch, name))
        .collect::<Result<_, _>>()?;
    let weight_column = find_float_column_from_candidates(batch, &["weight".to_string()])?;

    for row in 0..batch.num_rows() {
        for (target, source) in p4_columns_out.iter_mut().zip(&p4_columns) {
            target.px.push(source.px.value(row));
            target.py.push(source.py.value(row));
            target.pz.push(source.pz.value(row));
            target.e.push(source.e.value(row));
        }
        for (target, source) in aux_columns_out.iter_mut().zip(&aux_columns) {
            target.push(source.value(row));
        }
        weights_out.push(
            weight_column
                .as_ref()
                .map(|column| column.value(row))
                .unwrap_or(1.0),
        );
    }

    Ok(())
}

#[cfg(feature = "mpi")]
fn trim_columnar_rows(columnar: &mut DatasetStorage, drop_front: usize, expected_len: usize) {
    if drop_front > 0 {
        for column in &mut columnar.p4 {
            column.px.drain(0..drop_front);
            column.py.drain(0..drop_front);
            column.pz.drain(0..drop_front);
            column.e.drain(0..drop_front);
        }
        for column in &mut columnar.aux {
            column.drain(0..drop_front);
        }
        columnar.weights.drain(0..drop_front);
    }

    if columnar.n_events() > expected_len {
        for column in &mut columnar.p4 {
            column.px.truncate(expected_len);
            column.py.truncate(expected_len);
            column.pz.truncate(expected_len);
            column.e.truncate(expected_len);
        }
        for column in &mut columnar.aux {
            column.truncate(expected_len);
        }
        columnar.weights.truncate(expected_len);
    }
}

/// Load a [`Dataset`] from a ROOT TTree using the oxyroot backend.
pub fn read_root(file_path: &str, options: &DatasetReadOptions) -> LadduResult<Arc<Dataset>> {
    let storage = read_root_storage(file_path, options)?;
    Ok(Arc::new(storage.to_dataset()))
}

/// Load a [`DatasetStorage`] from a ROOT TTree without materializing row-event storage.
pub(crate) fn read_root_storage(
    file_path: &str,
    options: &DatasetReadOptions,
) -> LadduResult<Arc<DatasetStorage>> {
    let root_data = read_root_columns(file_path, options)?;
    let p4 = root_data
        .p4_columns
        .into_iter()
        .map(|columns| ColumnarP4Column {
            px: columns.px,
            py: columns.py,
            pz: columns.pz,
            e: columns.e,
        })
        .collect::<Vec<_>>();
    Ok(Arc::new(DatasetStorage {
        metadata: root_data.metadata,
        p4,
        aux: root_data.aux_columns,
        weights: root_data.weight_values,
    }))
}

struct RootP4Columns {
    px: Vec<f64>,
    py: Vec<f64>,
    pz: Vec<f64>,
    e: Vec<f64>,
}

struct RootReadColumns {
    metadata: Arc<DatasetMetadata>,
    p4_columns: Vec<RootP4Columns>,
    aux_columns: Vec<Vec<f64>>,
    weight_values: Vec<f64>,
    n_events: usize,
}

fn read_root_columns(
    file_path: &str,
    options: &DatasetReadOptions,
) -> LadduResult<RootReadColumns> {
    let path = canonicalize_dataset_path(file_path)?;
    let mut file = RootFile::open(&path).map_err(|err| {
        LadduError::Custom(format!(
            "Failed to open ROOT file '{}': {err}",
            path.display()
        ))
    })?;

    let (tree, tree_name) = resolve_root_tree(&mut file, options.tree.as_deref())?;

    let branches: Vec<&Branch> = tree.branches().collect();
    let mut lookup: BranchLookup<'_> = IndexMap::new();
    for &branch in &branches {
        if let Some(kind) = branch_scalar_kind(branch) {
            lookup.insert(branch.name(), (kind, branch));
        }
    }

    if lookup.is_empty() {
        return Err(LadduError::Custom(format!(
            "No float or double branches found in ROOT tree '{tree_name}'"
        )));
    }

    let column_names: Vec<&str> = lookup.keys().copied().collect();
    let (detected_p4_names, detected_aux_names) = infer_p4_and_aux_names(&column_names);
    let metadata = options.resolve_metadata(detected_p4_names, detected_aux_names)?;

    // TODO: do all reads in parallel if possible to match parquet impl
    let mut p4_columns = Vec::with_capacity(metadata.p4_names.len());
    for name in &metadata.p4_names {
        let logical = format!("{name}_px");
        let px = read_branch_values_from_candidates(
            &lookup,
            &component_candidates(name, "px"),
            &logical,
        )?;

        let logical = format!("{name}_py");
        let py = read_branch_values_from_candidates(
            &lookup,
            &component_candidates(name, "py"),
            &logical,
        )?;

        let logical = format!("{name}_pz");
        let pz = read_branch_values_from_candidates(
            &lookup,
            &component_candidates(name, "pz"),
            &logical,
        )?;

        let logical = format!("{name}_e");
        let e = read_branch_values_from_candidates(
            &lookup,
            &component_candidates(name, "e"),
            &logical,
        )?;

        p4_columns.push(RootP4Columns { px, py, pz, e });
    }

    let mut aux_columns = Vec::with_capacity(metadata.aux_names.len());
    for name in &metadata.aux_names {
        let values = read_branch_values(&lookup, name)?;
        aux_columns.push(values);
    }

    let n_events = if let Some(first) = p4_columns.first() {
        first.px.len()
    } else if let Some(first) = aux_columns.first() {
        first.len()
    } else {
        return Err(LadduError::Custom(
            "Unable to determine event count; dataset has no four-momentum or auxiliary columns"
                .to_string(),
        ));
    };

    let weight_values = match read_branch_values_optional(&lookup, "weight")? {
        Some(values) => {
            if values.len() != n_events {
                return Err(LadduError::LengthMismatch {
                    context: "Column 'weight'".to_string(),
                    expected: n_events,
                    actual: values.len(),
                });
            }
            values
        }
        None => vec![1.0; n_events],
    };

    for columns in &p4_columns {
        for (component_name, component) in [
            ("px", &columns.px),
            ("py", &columns.py),
            ("pz", &columns.pz),
            ("e", &columns.e),
        ] {
            if component.len() != n_events {
                return Err(LadduError::LengthMismatch {
                    context: format!("ROOT p4 component '{component_name}'"),
                    expected: n_events,
                    actual: component.len(),
                });
            }
        }
    }
    for (aux_index, column) in aux_columns.iter().enumerate() {
        if column.len() != n_events {
            return Err(LadduError::LengthMismatch {
                context: format!("ROOT aux column index {aux_index}"),
                expected: n_events,
                actual: column.len(),
            });
        }
    }

    Ok(RootReadColumns {
        metadata,
        p4_columns,
        aux_columns,
        weight_values,
        n_events,
    })
}

/// Persist a [`Dataset`] to a Parquet file.
pub fn write_parquet(
    dataset: &Dataset,
    file_path: &str,
    options: &DatasetWriteOptions,
) -> LadduResult<()> {
    write_parquet_storage(&dataset.columnar, file_path, options)
}

/// Persist a [`DatasetStorage`] to a Parquet file.
pub(crate) fn write_parquet_storage(
    dataset: &DatasetStorage,
    file_path: &str,
    options: &DatasetWriteOptions,
) -> LadduResult<()> {
    let path = expand_output_path(file_path)?;
    dataset.write_parquet_impl(path, options)
}

/// Persist a [`Dataset`] to a ROOT file using the oxyroot backend.
pub fn write_root(
    dataset: &Dataset,
    file_path: &str,
    options: &DatasetWriteOptions,
) -> LadduResult<()> {
    let path = expand_output_path(file_path)?;
    dataset.write_root_impl(path, options)
}

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

    fn resolve_metadata(
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

const P4_COMPONENT_SUFFIXES: [&str; 4] = ["_px", "_py", "_pz", "_e"];
const DEFAULT_WRITE_BATCH_SIZE: usize = 10_000;

fn detect_columns(file_path: &PathBuf) -> LadduResult<(Vec<String>, Vec<String>)> {
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let float_cols: Vec<&str> = schema
        .fields()
        .iter()
        .filter(|f| matches!(f.data_type(), DataType::Float32 | DataType::Float64))
        .map(|f| f.name().as_str())
        .collect();
    Ok(infer_p4_and_aux_names(&float_cols))
}

fn infer_p4_and_aux_names(float_cols: &[&str]) -> (Vec<String>, Vec<String>) {
    let suffix_set: IndexSet<&str> = P4_COMPONENT_SUFFIXES.iter().copied().collect();
    let mut groups: IndexMap<&str, IndexSet<&str>> = IndexMap::new();
    for col in float_cols {
        for suffix in &suffix_set {
            if let Some(prefix) = col.strip_suffix(suffix) {
                groups.entry(prefix).or_default().insert(*suffix);
            }
        }
    }

    let mut p4_names: Vec<String> = Vec::new();
    let mut p4_columns: IndexSet<String> = IndexSet::new();
    for (prefix, suffixes) in &groups {
        if suffixes.len() == suffix_set.len() {
            p4_names.push((*prefix).to_string());
            for suffix in &suffix_set {
                p4_columns.insert(format!("{prefix}{suffix}"));
            }
        }
    }

    let mut aux_names: Vec<String> = Vec::new();
    for col in float_cols {
        if p4_columns.contains(*col) {
            continue;
        }
        if col.eq_ignore_ascii_case("weight") {
            continue;
        }
        aux_names.push((*col).to_string());
    }

    (p4_names, aux_names)
}

type BranchLookup<'a> = IndexMap<&'a str, (RootScalarKind, &'a Branch)>;

#[derive(Clone, Copy)]
enum RootScalarKind {
    F32,
    F64,
}

fn branch_scalar_kind(branch: &Branch) -> Option<RootScalarKind> {
    let type_name = branch.item_type_name();
    let lower = type_name.to_ascii_lowercase();
    if lower.contains("vector") {
        return None;
    }
    match lower.as_str() {
        "float" | "float_t" | "float32_t" => Some(RootScalarKind::F32),
        "double" | "double_t" | "double32_t" => Some(RootScalarKind::F64),
        _ => None,
    }
}

fn read_branch_values<'a>(lookup: &BranchLookup<'a>, column_name: &str) -> LadduResult<Vec<f64>> {
    let (kind, branch) =
        lookup
            .get(column_name)
            .copied()
            .ok_or_else(|| LadduError::MissingColumn {
                name: column_name.to_string(),
            })?;

    let values = match kind {
        RootScalarKind::F32 => branch
            .as_iter::<f32>()
            .map_err(|err| map_root_error(&format!("Failed to read branch '{column_name}'"), err))?
            .map(|value| value as f64)
            .collect(),
        RootScalarKind::F64 => branch
            .as_iter::<f64>()
            .map_err(|err| map_root_error(&format!("Failed to read branch '{column_name}'"), err))?
            .collect(),
    };
    Ok(values)
}

fn read_branch_values_optional<'a>(
    lookup: &BranchLookup<'a>,
    column_name: &str,
) -> LadduResult<Option<Vec<f64>>> {
    if lookup.contains_key(column_name) {
        read_branch_values(lookup, column_name).map(Some)
    } else {
        Ok(None)
    }
}

fn read_branch_values_from_candidates<'a>(
    lookup: &BranchLookup<'a>,
    candidates: &[String],
    logical_name: &str,
) -> LadduResult<Vec<f64>> {
    for candidate in candidates {
        if lookup.contains_key(candidate.as_str()) {
            return read_branch_values(lookup, candidate);
        }
    }
    Err(LadduError::MissingColumn {
        name: logical_name.to_string(),
    })
}

fn resolve_root_tree(
    file: &mut RootFile,
    requested: Option<&str>,
) -> LadduResult<(ReaderTree, String)> {
    if let Some(name) = requested {
        let tree = file
            .get_tree(name)
            .map_err(|err| map_root_error(&format!("Failed to open ROOT tree '{name}'"), err))?;
        return Ok((tree, name.to_string()));
    }

    let tree_names: Vec<String> = file
        .keys()
        .into_iter()
        .filter(|key| key.class_name() == "TTree")
        .map(|key| key.name().to_string())
        .collect();

    if tree_names.is_empty() {
        return Err(LadduError::Custom(
            "ROOT file does not contain any TTrees".to_string(),
        ));
    }

    if tree_names.len() > 1 {
        return Err(LadduError::Custom(format!(
            "Multiple TTrees found ({:?}); specify DatasetReadOptions::tree to disambiguate",
            tree_names
        )));
    }

    let selected = &tree_names[0];
    let tree = file
        .get_tree(selected)
        .map_err(|err| map_root_error(&format!("Failed to open ROOT tree '{selected}'"), err))?;
    Ok((tree, selected.clone()))
}

fn map_root_error<E: std::fmt::Display>(context: &str, err: E) -> LadduError {
    LadduError::Custom(format!("{context}: {err}")) // NOTE: the oxyroot error type is not public
}

#[derive(Clone, Copy)]
enum FloatColumn<'a> {
    F32(&'a Float32Array),
    F64(&'a Float64Array),
}

impl<'a> FloatColumn<'a> {
    fn value(&self, row: usize) -> f64 {
        match self {
            Self::F32(array) => array.value(row) as f64,
            Self::F64(array) => array.value(row),
        }
    }
}

struct P4Columns<'a> {
    px: FloatColumn<'a>,
    py: FloatColumn<'a>,
    pz: FloatColumn<'a>,
    e: FloatColumn<'a>,
}

fn prepare_float_column<'a>(batch: &'a RecordBatch, name: &str) -> LadduResult<FloatColumn<'a>> {
    prepare_float_column_from_candidates(batch, &[name.to_string()], name)
}

fn prepare_p4_columns<'a>(batch: &'a RecordBatch, name: &str) -> LadduResult<P4Columns<'a>> {
    Ok(P4Columns {
        px: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "px"),
            &format!("{name}_px"),
        )?,
        py: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "py"),
            &format!("{name}_py"),
        )?,
        pz: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "pz"),
            &format!("{name}_pz"),
        )?,
        e: prepare_float_column_from_candidates(
            batch,
            &component_candidates(name, "e"),
            &format!("{name}_e"),
        )?,
    })
}

fn component_candidates(name: &str, suffix: &str) -> Vec<String> {
    let mut candidates = Vec::with_capacity(3);
    let base = format!("{name}_{suffix}");
    candidates.push(base.clone());

    let mut capitalized = suffix.to_string();
    if let Some(first) = capitalized.get_mut(0..1) {
        first.make_ascii_uppercase();
    }
    if capitalized != suffix {
        candidates.push(format!("{name}_{capitalized}"));
    }

    let upper = suffix.to_ascii_uppercase();
    if upper != suffix && upper != capitalized {
        candidates.push(format!("{name}_{upper}"));
    }

    candidates
}

fn find_float_column_from_candidates<'a>(
    batch: &'a RecordBatch,
    candidates: &[String],
) -> LadduResult<Option<FloatColumn<'a>>> {
    use arrow::datatypes::DataType;

    for candidate in candidates {
        if let Some(column) = batch.column_by_name(candidate) {
            return match column.data_type() {
                DataType::Float32 => Ok(Some(FloatColumn::F32(
                    column
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("Column advertised as Float32 but could not be downcast"),
                ))),
                DataType::Float64 => Ok(Some(FloatColumn::F64(
                    column
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("Column advertised as Float64 but could not be downcast"),
                ))),
                other => {
                    return Err(LadduError::InvalidColumnType {
                        name: candidate.clone(),
                        datatype: other.to_string(),
                    });
                }
            };
        }
    }
    Ok(None)
}

fn prepare_float_column_from_candidates<'a>(
    batch: &'a RecordBatch,
    candidates: &[String],
    logical_name: &str,
) -> LadduResult<FloatColumn<'a>> {
    find_float_column_from_candidates(batch, candidates)?.ok_or_else(|| LadduError::MissingColumn {
        name: logical_name.to_string(),
    })
}

fn record_batch_to_events(
    batch: RecordBatch,
    p4_names: &[String],
    aux_names: &[String],
) -> LadduResult<Vec<Arc<EventData>>> {
    let batch_ref = &batch;
    let p4_columns: Vec<P4Columns<'_>> = p4_names
        .iter()
        .map(|name| prepare_p4_columns(batch_ref, name))
        .collect::<Result<_, _>>()?;

    let aux_columns: Vec<FloatColumn<'_>> = aux_names
        .iter()
        .map(|name| prepare_float_column(batch_ref, name))
        .collect::<Result<_, _>>()?;

    let weight_column = find_float_column_from_candidates(batch_ref, &["weight".to_string()])?;

    let mut events = Vec::with_capacity(batch_ref.num_rows());
    for row in 0..batch_ref.num_rows() {
        let mut p4s = Vec::with_capacity(p4_columns.len());
        for columns in &p4_columns {
            let px = columns.px.value(row);
            let py = columns.py.value(row);
            let pz = columns.pz.value(row);
            let e = columns.e.value(row);
            p4s.push(Vec4::new(px, py, pz, e));
        }

        let mut aux = Vec::with_capacity(aux_columns.len());
        for column in &aux_columns {
            aux.push(column.value(row));
        }

        let event = EventData {
            p4s,
            aux,
            weight: weight_column
                .as_ref()
                .map(|column| column.value(row))
                .unwrap_or(1.0),
        };
        events.push(Arc::new(event));
    }
    Ok(events)
}

fn columnar_range_to_record_batch(
    dataset: &DatasetStorage,
    start: usize,
    end: usize,
    schema: Arc<Schema>,
    precision: FloatPrecision,
) -> arrow::error::Result<RecordBatch> {
    let mut columns: Vec<arrow::array::ArrayRef> = Vec::new();
    match precision {
        FloatPrecision::F64 => {
            for p4 in &dataset.p4 {
                columns.push(Arc::new(Float64Array::from(p4.px[start..end].to_vec())));
                columns.push(Arc::new(Float64Array::from(p4.py[start..end].to_vec())));
                columns.push(Arc::new(Float64Array::from(p4.pz[start..end].to_vec())));
                columns.push(Arc::new(Float64Array::from(p4.e[start..end].to_vec())));
            }
            for aux in &dataset.aux {
                columns.push(Arc::new(Float64Array::from(aux[start..end].to_vec())));
            }
            columns.push(Arc::new(Float64Array::from(
                dataset.weights[start..end].to_vec(),
            )));
        }
        FloatPrecision::F32 => {
            for p4 in &dataset.p4 {
                columns.push(Arc::new(Float32Array::from(
                    p4.px[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
                columns.push(Arc::new(Float32Array::from(
                    p4.py[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
                columns.push(Arc::new(Float32Array::from(
                    p4.pz[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
                columns.push(Arc::new(Float32Array::from(
                    p4.e[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
            }
            for aux in &dataset.aux {
                columns.push(Arc::new(Float32Array::from(
                    aux[start..end]
                        .iter()
                        .map(|v| *v as f32)
                        .collect::<Vec<_>>(),
                )));
            }
            columns.push(Arc::new(Float32Array::from(
                dataset.weights[start..end]
                    .iter()
                    .map(|v| *v as f32)
                    .collect::<Vec<_>>(),
            )));
        }
    }
    RecordBatch::try_new(schema, columns)
}

fn build_parquet_schema(metadata: &DatasetMetadata, precision: FloatPrecision) -> Schema {
    let dtype = match precision {
        FloatPrecision::F64 => DataType::Float64,
        FloatPrecision::F32 => DataType::Float32,
    };

    let mut fields = Vec::new();
    for name in &metadata.p4_names {
        for suffix in P4_COMPONENT_SUFFIXES {
            fields.push(Field::new(format!("{name}{suffix}"), dtype.clone(), false));
        }
    }

    for name in &metadata.aux_names {
        fields.push(Field::new(name.clone(), dtype.clone(), false));
    }

    fields.push(Field::new("weight", dtype, false));
    Schema::new(fields)
}

trait FromF64 {
    fn from_f64(value: f64) -> Self;
}

impl FromF64 for f64 {
    fn from_f64(value: f64) -> Self {
        value
    }
}

impl FromF64 for f32 {
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

struct SharedEventFetcher {
    dataset: Arc<Dataset>,
    world: Option<WorldHandle>,
    total: usize,
    branch_count: usize,
    current_index: Option<usize>,
    current_event: Option<Event>,
    remaining: usize,
}

impl SharedEventFetcher {
    fn new(
        dataset: Arc<Dataset>,
        world: Option<WorldHandle>,
        total: usize,
        branch_count: usize,
    ) -> Self {
        Self {
            dataset,
            world,
            total,
            branch_count,
            current_index: None,
            current_event: None,
            remaining: 0,
        }
    }

    fn event_for_index(&mut self, index: usize) -> Option<Event> {
        if index >= self.total {
            return None;
        }

        let refresh_needed = match self.current_index {
            None => true,
            Some(current) => current != index || self.remaining == 0,
        };

        if refresh_needed {
            let event =
                fetch_event_for_index(&self.dataset, index, self.total, self.world.as_ref());
            self.current_index = Some(index);
            self.remaining = self.branch_count;
            self.current_event = Some(event);
        }

        let event = self.current_event.as_ref().cloned();
        if self.remaining > 0 {
            self.remaining -= 1;
        }
        if self.remaining == 0 {
            // Drop the cached event so the next request fetches the next index.
            self.current_event = None;
        }
        event
    }
}

enum ColumnKind {
    Px(usize),
    Py(usize),
    Pz(usize),
    E(usize),
    Aux(usize),
    Weight,
}

struct ColumnIterator<T> {
    fetcher: Arc<Mutex<SharedEventFetcher>>,
    index: usize,
    kind: ColumnKind,
    _marker: std::marker::PhantomData<T>,
}

impl<T> ColumnIterator<T> {
    fn new(fetcher: Arc<Mutex<SharedEventFetcher>>, kind: ColumnKind) -> Self {
        Self {
            fetcher,
            index: 0,
            kind,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> Iterator for ColumnIterator<T>
where
    T: FromF64,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut fetcher = self.fetcher.lock();
        let event = fetcher.event_for_index(self.index)?;
        self.index += 1;

        match self.kind {
            ColumnKind::Px(idx) => event.p4s.get(idx).map(|p4| T::from_f64(p4.x)),
            ColumnKind::Py(idx) => event.p4s.get(idx).map(|p4| T::from_f64(p4.y)),
            ColumnKind::Pz(idx) => event.p4s.get(idx).map(|p4| T::from_f64(p4.z)),
            ColumnKind::E(idx) => event.p4s.get(idx).map(|p4| T::from_f64(p4.t)),
            ColumnKind::Aux(idx) => event.aux.get(idx).map(|value| T::from_f64(*value)),
            ColumnKind::Weight => Some(T::from_f64(event.weight)),
        }
    }
}

fn build_root_column_iterators<T>(
    dataset: Arc<Dataset>,
    world: Option<WorldHandle>,
    branch_count: usize,
    total: usize,
) -> Vec<(String, ColumnIterator<T>)>
where
    T: FromF64,
{
    let fetcher = Arc::new(Mutex::new(SharedEventFetcher::new(
        dataset,
        world,
        total,
        branch_count,
    )));

    let p4_names: Vec<String> = fetcher.lock().dataset.metadata.p4_names.clone();
    let aux_names: Vec<String> = fetcher.lock().dataset.metadata.aux_names.clone();

    let mut iterators = Vec::new();

    for (idx, name) in p4_names.iter().enumerate() {
        iterators.push((
            format!("{name}_px"),
            ColumnIterator::new(fetcher.clone(), ColumnKind::Px(idx)),
        ));
        iterators.push((
            format!("{name}_py"),
            ColumnIterator::new(fetcher.clone(), ColumnKind::Py(idx)),
        ));
        iterators.push((
            format!("{name}_pz"),
            ColumnIterator::new(fetcher.clone(), ColumnKind::Pz(idx)),
        ));
        iterators.push((
            format!("{name}_e"),
            ColumnIterator::new(fetcher.clone(), ColumnKind::E(idx)),
        ));
    }

    for (idx, name) in aux_names.iter().enumerate() {
        iterators.push((
            name.clone(),
            ColumnIterator::new(fetcher.clone(), ColumnKind::Aux(idx)),
        ));
    }

    iterators.push((
        "weight".to_string(),
        ColumnIterator::new(fetcher, ColumnKind::Weight),
    ));

    iterators
}

fn drain_column_iterators<T>(iterators: &mut [(String, ColumnIterator<T>)], n_events: usize)
where
    T: FromF64,
{
    for _ in 0..n_events {
        for (_name, iterator) in iterators.iter_mut() {
            let _ = iterator.next();
        }
    }
}

fn fetch_event_for_index(
    dataset: &Dataset,
    index: usize,
    total: usize,
    world: Option<&WorldHandle>,
) -> Event {
    let _ = total;
    let _ = world;
    #[cfg(feature = "mpi")]
    {
        if let Some(world) = world {
            return fetch_event_mpi(dataset, index, world, total);
        }
    }

    dataset.index_local(index).clone()
}

impl Dataset {
    #[allow(clippy::too_many_arguments)]
    fn write_root_with_type<T>(
        &self,
        dataset: Arc<Dataset>,
        world: Option<WorldHandle>,
        is_root: bool,
        file_path: &Path,
        tree_name: &str,
        branch_count: usize,
        total_events: usize,
    ) -> LadduResult<()>
    where
        T: FromF64 + oxyroot::Marshaler + 'static,
    {
        let mut iterators =
            build_root_column_iterators::<T>(dataset, world, branch_count, total_events);

        if is_root {
            let mut file = RootFile::create(file_path).map_err(|err| {
                LadduError::Custom(format!(
                    "Failed to create ROOT file '{}': {err}",
                    file_path.display()
                ))
            })?;

            let mut tree = WriterTree::new(tree_name);
            for (name, iterator) in iterators {
                tree.new_branch(name, iterator);
            }

            tree.write(&mut file).map_err(|err| {
                LadduError::Custom(format!(
                    "Failed to write ROOT tree '{tree_name}' to '{}': {err}",
                    file_path.display()
                ))
            })?;

            file.close().map_err(|err| {
                LadduError::Custom(format!(
                    "Failed to close ROOT file '{}': {err}",
                    file_path.display()
                ))
            })?;
        } else {
            drain_column_iterators(&mut iterators, total_events);
        }

        Ok(())
    }
}

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

#[cfg(test)]
mod tests {
    use crate::Mass;

    use super::*;
    #[cfg(feature = "mpi")]
    use crate::mpi::{finalize_mpi, get_world, use_mpi};
    use crate::utils::vectors::Vec3;
    use approx::{assert_relative_eq, assert_relative_ne};
    use fastrand;
    use serde::{Deserialize, Serialize};
    use std::{
        env, fs,
        path::{Path, PathBuf},
    };

    fn test_data_path(file: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join(file)
    }

    fn open_test_dataset(file: &str, options: DatasetReadOptions) -> Arc<Dataset> {
        let path = test_data_path(file);
        let path_str = path.to_str().expect("test data path should be valid UTF-8");
        let ext = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        match ext.as_str() {
            "parquet" => read_parquet(path_str, &options),
            "root" => read_root(path_str, &options),
            other => panic!("Unsupported extension in test data: {other}"),
        }
        .expect("dataset should open")
    }

    fn make_temp_dir() -> PathBuf {
        let dir = env::temp_dir().join(format!("laddu_test_{}", fastrand::u64(..)));
        fs::create_dir(&dir).expect("temp dir should be created");
        dir
    }

    fn assert_events_close(left: &Event, right: &Event, p4_names: &[&str], aux_names: &[&str]) {
        for name in p4_names {
            let lp4 = left
                .p4(name)
                .unwrap_or_else(|| panic!("missing p4 '{name}' in left dataset"));
            let rp4 = right
                .p4(name)
                .unwrap_or_else(|| panic!("missing p4 '{name}' in right dataset"));
            assert_relative_eq!(lp4.px(), rp4.px(), epsilon = 1e-9);
            assert_relative_eq!(lp4.py(), rp4.py(), epsilon = 1e-9);
            assert_relative_eq!(lp4.pz(), rp4.pz(), epsilon = 1e-9);
            assert_relative_eq!(lp4.e(), rp4.e(), epsilon = 1e-9);
        }
        let left_aux = left.aux();
        let right_aux = right.aux();
        for name in aux_names {
            let laux = left_aux
                .get(name)
                .copied()
                .unwrap_or_else(|| panic!("missing aux '{name}' in left dataset"));
            let raux = right_aux
                .get(name)
                .copied()
                .unwrap_or_else(|| panic!("missing aux '{name}' in right dataset"));
            assert_relative_eq!(laux, raux, epsilon = 1e-9);
        }
        assert_relative_eq!(left.weight(), right.weight(), epsilon = 1e-9);
    }

    fn assert_datasets_close(
        left: &Arc<Dataset>,
        right: &Arc<Dataset>,
        p4_names: &[&str],
        aux_names: &[&str],
    ) {
        assert_eq!(left.n_events(), right.n_events());
        for idx in 0..left.n_events() {
            let Ok(levent) = left.event(idx) else {
                panic!("left dataset missing event at index {idx}");
            };
            let Ok(revent) = right.event(idx) else {
                panic!("right dataset missing event at index {idx}");
            };
            assert_events_close(&levent, &revent, p4_names, aux_names);
        }
    }

    fn assert_dataset_columnar_close(left: &DatasetStorage, right: &DatasetStorage) {
        assert_eq!(left.n_events(), right.n_events());
        assert_eq!(left.metadata().p4_names(), right.metadata().p4_names());
        assert_eq!(left.metadata().aux_names(), right.metadata().aux_names());
        for event_index in 0..left.n_events() {
            for p4_index in 0..left.metadata().p4_names().len() {
                let lp4 = left.p4(event_index, p4_index);
                let rp4 = right.p4(event_index, p4_index);
                assert_relative_eq!(lp4.px(), rp4.px(), epsilon = 1e-12);
                assert_relative_eq!(lp4.py(), rp4.py(), epsilon = 1e-12);
                assert_relative_eq!(lp4.pz(), rp4.pz(), epsilon = 1e-12);
                assert_relative_eq!(lp4.e(), rp4.e(), epsilon = 1e-12);
            }
            for aux_index in 0..left.metadata().aux_names().len() {
                let l = left.aux(event_index, aux_index);
                let r = right.aux(event_index, aux_index);
                assert_relative_eq!(l, r, epsilon = 1e-12);
            }
            let lw = left.weight(event_index);
            let rw = right.weight(event_index);
            assert_relative_eq!(lw, rw, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_from_parquet_auto_matches_explicit_names() {
        let auto = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
        let explicit_options = DatasetReadOptions::new()
            .p4_names(TEST_P4_NAMES)
            .aux_names(TEST_AUX_NAMES);
        let explicit = open_test_dataset("data_f32.parquet", explicit_options);

        let mut detected_p4: Vec<&str> = auto.p4_names().iter().map(String::as_str).collect();
        detected_p4.sort_unstable();
        let mut expected_p4 = TEST_P4_NAMES.to_vec();
        expected_p4.sort_unstable();
        assert_eq!(detected_p4, expected_p4);
        let mut detected_aux: Vec<&str> = auto.aux_names().iter().map(String::as_str).collect();
        detected_aux.sort_unstable();
        let mut expected_aux = TEST_AUX_NAMES.to_vec();
        expected_aux.sort_unstable();
        assert_eq!(detected_aux, expected_aux);
        assert_datasets_close(&auto, &explicit, TEST_P4_NAMES, TEST_AUX_NAMES);
    }

    #[test]
    fn test_from_parquet_with_aliases() {
        let dataset = open_test_dataset(
            "data_f32.parquet",
            DatasetReadOptions::new().alias("resonance", ["kshort1", "kshort2"]),
        );
        let event = dataset.named_event(0).expect("event should exist");
        let alias_vec = event.p4("resonance").expect("alias vector");
        let expected = event.get_p4_sum(["kshort1", "kshort2"]);
        assert_relative_eq!(alias_vec.px(), expected.px(), epsilon = 1e-9);
        assert_relative_eq!(alias_vec.py(), expected.py(), epsilon = 1e-9);
        assert_relative_eq!(alias_vec.pz(), expected.pz(), epsilon = 1e-9);
        assert_relative_eq!(alias_vec.e(), expected.e(), epsilon = 1e-9);
    }

    #[test]
    fn test_from_parquet_f64_matches_f32() {
        let f32_ds = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
        let f64_ds = open_test_dataset("data_f64.parquet", DatasetReadOptions::new());
        assert_datasets_close(&f64_ds, &f32_ds, TEST_P4_NAMES, TEST_AUX_NAMES);
    }

    #[test]
    fn test_from_root_detects_columns_and_matches_parquet() {
        let parquet = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
        let root_auto = open_test_dataset("data_f32.root", DatasetReadOptions::new());
        let mut detected_p4: Vec<&str> = root_auto.p4_names().iter().map(String::as_str).collect();
        detected_p4.sort_unstable();
        let mut expected_p4 = TEST_P4_NAMES.to_vec();
        expected_p4.sort_unstable();
        assert_eq!(detected_p4, expected_p4);
        let mut detected_aux: Vec<&str> =
            root_auto.aux_names().iter().map(String::as_str).collect();
        detected_aux.sort_unstable();
        let mut expected_aux = TEST_AUX_NAMES.to_vec();
        expected_aux.sort_unstable();
        assert_eq!(detected_aux, expected_aux);
        let root_named_options = DatasetReadOptions::new()
            .p4_names(TEST_P4_NAMES)
            .aux_names(TEST_AUX_NAMES);
        let root_named = open_test_dataset("data_f32.root", root_named_options);
        assert_datasets_close(&root_auto, &root_named, TEST_P4_NAMES, TEST_AUX_NAMES);
        assert_datasets_close(&root_auto, &parquet, TEST_P4_NAMES, TEST_AUX_NAMES);
    }

    #[test]
    fn test_from_root_f64_matches_parquet() {
        let parquet = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
        let root_f64 = open_test_dataset("data_f64.root", DatasetReadOptions::new());
        assert_datasets_close(&root_f64, &parquet, TEST_P4_NAMES, TEST_AUX_NAMES);
    }
    #[test]
    fn test_event_creation() {
        let event = test_event();
        assert_eq!(event.p4s.len(), 4);
        assert_eq!(event.aux.len(), 2);
        assert_relative_eq!(event.weight, 0.48)
    }

    #[test]
    fn test_event_p4_sum() {
        let event = test_event();
        let sum = event.get_p4_sum([2, 3]);
        assert_relative_eq!(sum.px(), event.p4s[2].px() + event.p4s[3].px());
        assert_relative_eq!(sum.py(), event.p4s[2].py() + event.p4s[3].py());
        assert_relative_eq!(sum.pz(), event.p4s[2].pz() + event.p4s[3].pz());
        assert_relative_eq!(sum.e(), event.p4s[2].e() + event.p4s[3].e());
    }

    #[test]
    fn test_event_boost() {
        let event = test_event();
        let event_boosted = event.boost_to_rest_frame_of([1, 2, 3]);
        let p4_sum = event_boosted.get_p4_sum([1, 2, 3]);
        assert_relative_eq!(p4_sum.px(), 0.0);
        assert_relative_eq!(p4_sum.py(), 0.0);
        assert_relative_eq!(p4_sum.pz(), 0.0, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn test_event_evaluate() {
        let event = test_event();
        let mut mass = Mass::new(["proton"]);
        mass.bind(
            &DatasetMetadata::new(
                TEST_P4_NAMES.iter().map(|s| (*s).to_string()).collect(),
                TEST_AUX_NAMES.iter().map(|s| (*s).to_string()).collect(),
            )
            .expect("metadata"),
        )
        .unwrap();
        assert_relative_eq!(event.evaluate(&mass), 1.007);
    }

    #[test]
    fn test_dataset_size_check() {
        let dataset = Dataset::new(Vec::new());
        assert_eq!(dataset.n_events(), 0);
        let dataset = Dataset::new(vec![Arc::new(test_event())]);
        assert_eq!(dataset.n_events(), 1);
    }

    #[test]
    fn test_dataset_sum() {
        let dataset = test_dataset();
        let metadata = dataset.metadata_arc();
        let dataset2 = Dataset::new_with_metadata(
            vec![Arc::new(EventData {
                p4s: test_event().p4s,
                aux: test_event().aux,
                weight: 0.52,
            })],
            metadata.clone(),
        );
        let dataset_sum = &dataset + &dataset2;
        assert_eq!(
            dataset_sum.event(0).expect("event should exist").weight,
            dataset.event(0).expect("event should exist").weight
        );
        assert_eq!(
            dataset_sum.event(1).expect("event should exist").weight,
            dataset2.event(0).expect("event should exist").weight
        );
    }

    #[test]
    fn test_dataset_weights() {
        let dataset = Dataset::new(vec![
            Arc::new(test_event()),
            Arc::new(EventData {
                p4s: test_event().p4s,
                aux: test_event().aux,
                weight: 0.52,
            }),
        ]);
        let weights = dataset.weights();
        assert_eq!(weights.len(), 2);
        assert_relative_eq!(weights[0], 0.48);
        assert_relative_eq!(weights[1], 0.52);
        assert_relative_eq!(dataset.n_events_weighted(), 1.0);
    }

    #[test]
    fn test_dataset_columnar_matches_row_storage() {
        let dataset = Dataset::new(vec![
            Arc::new(test_event()),
            Arc::new(EventData {
                p4s: test_event().p4s,
                aux: test_event().aux,
                weight: 0.52,
            }),
        ]);
        let columnar = dataset
            .to_storage()
            .expect("columnar conversion should succeed");
        assert_eq!(columnar.n_events(), dataset.n_events_local());
        for event_index in 0..dataset.n_events_local() {
            let row = dataset.event(event_index).expect("event should exist");
            for p4_index in 0..row.p4s.len() {
                let p4 = columnar.p4(event_index, p4_index);
                assert_relative_eq!(p4.px(), row.p4s[p4_index].px(), epsilon = 1e-12);
                assert_relative_eq!(p4.py(), row.p4s[p4_index].py(), epsilon = 1e-12);
                assert_relative_eq!(p4.pz(), row.p4s[p4_index].pz(), epsilon = 1e-12);
                assert_relative_eq!(p4.e(), row.p4s[p4_index].e(), epsilon = 1e-12);
            }
            for aux_index in 0..row.aux.len() {
                assert_relative_eq!(
                    columnar.aux(event_index, aux_index),
                    row.aux[aux_index],
                    epsilon = 1e-12
                );
            }
            assert_relative_eq!(columnar.weight(event_index), row.weight, epsilon = 1e-12);
            let rebuilt = columnar.event_data(event_index);
            assert_eq!(rebuilt.p4s.len(), row.p4s.len());
            assert_eq!(rebuilt.aux.len(), row.aux.len());
            assert_relative_eq!(rebuilt.weight, row.weight, epsilon = 1e-12);
        }
    }

    #[test]
    #[should_panic(
        expected = "Dataset requires rectangular p4/aux columns for canonical columnar storage"
    )]
    fn test_dataset_rejects_ragged_rows_at_construction() {
        let _ = Dataset::new(vec![
            Arc::new(EventData {
                p4s: vec![Vec4::new(0.0, 0.0, 1.0, 1.0)],
                aux: vec![0.1],
                weight: 1.0,
            }),
            Arc::new(EventData {
                p4s: vec![],
                aux: vec![0.2, 0.3],
                weight: 2.0,
            }),
        ]);
    }

    #[test]
    fn test_dataset_columnar_named_row_view_matches_event_accessors() {
        let dataset = test_dataset();
        let columnar = dataset
            .to_storage()
            .expect("columnar conversion should succeed");
        let row = columnar.event_view(0);
        let named = dataset.named_event(0).expect("named event should exist");

        let beam = row.p4("beam").expect("beam p4 should be present");
        let named_beam = named.p4("beam").expect("named beam p4 should be present");
        assert_relative_eq!(beam.px(), named_beam.px(), epsilon = 1e-12);
        assert_relative_eq!(beam.py(), named_beam.py(), epsilon = 1e-12);
        assert_relative_eq!(beam.pz(), named_beam.pz(), epsilon = 1e-12);
        assert_relative_eq!(beam.e(), named_beam.e(), epsilon = 1e-12);

        assert_relative_eq!(
            row.aux("pol_angle").expect("pol angle should be present"),
            named.aux().get("pol_angle").copied().expect("named aux"),
            epsilon = 1e-12
        );
        assert_relative_eq!(row.weight(), named.weight(), epsilon = 1e-12);

        let summed = row
            .get_p4_sum(["kshort1", "kshort2"])
            .expect("named p4 sum should be present");
        let named_sum = named.get_p4_sum(["kshort1", "kshort2"]);
        assert_relative_eq!(summed.e(), named_sum.e(), epsilon = 1e-12);
    }

    #[test]
    fn test_dataset_columnar_row_view_variable_evaluation_matches_event_data() {
        let dataset = test_dataset();
        let columnar = dataset
            .to_storage()
            .expect("columnar conversion should succeed");
        let row = columnar.event_view(0);
        let named = dataset.named_event(0).expect("event should exist");
        let mut mass = Mass::new(["proton"]);
        mass.bind(dataset.metadata())
            .expect("mass should bind to metadata");
        let from_view = row.evaluate(&mass);
        let from_access_columnar = mass.value_view(&row);
        let from_event = dataset
            .event(0)
            .expect("event should exist")
            .evaluate(&mass);
        assert_relative_eq!(from_view, from_event, epsilon = 1e-12);
        assert_relative_eq!(from_access_columnar, from_event, epsilon = 1e-12);
        let from_named = named.evaluate(&mass);
        assert_relative_eq!(from_named, from_event, epsilon = 1e-12);
    }

    #[test]
    fn test_event_access_trait_on_event_data() {
        let event = test_event();
        assert_eq!(EventAccess::n_p4(&event), event.p4s.len());
        assert_eq!(EventAccess::n_aux(&event), event.aux.len());
        let proton = EventAccess::p4_at(&event, 1).expect("p4 at index should be available");
        assert_relative_eq!(proton.e(), event.p4s[1].e(), epsilon = 1e-12);
        let angle = EventAccess::aux_at(&event, 1).expect("aux at index should be available");
        assert_relative_eq!(angle, event.aux[1], epsilon = 1e-12);
        assert_relative_eq!(EventAccess::weight(&event), event.weight, epsilon = 1e-12);
    }

    #[test]
    fn test_named_event_access_trait_matches_aos_and_columnar_views() {
        let dataset = test_dataset();
        let aos = dataset.named_event(0).expect("event should exist");
        let columnar_storage = dataset
            .to_storage()
            .expect("columnar conversion should succeed");
        let columnar = columnar_storage.event_view(0);

        let aos_beam =
            NamedEventAccess::p4_by_name(&aos, "beam").expect("aos beam should be present");
        let columnar_beam = NamedEventAccess::p4_by_name(&columnar, "beam")
            .expect("columnar beam should be present");
        assert_relative_eq!(aos_beam.px(), columnar_beam.px(), epsilon = 1e-12);
        assert_relative_eq!(aos_beam.py(), columnar_beam.py(), epsilon = 1e-12);
        assert_relative_eq!(aos_beam.pz(), columnar_beam.pz(), epsilon = 1e-12);
        assert_relative_eq!(aos_beam.e(), columnar_beam.e(), epsilon = 1e-12);

        let aos_aux =
            NamedEventAccess::aux_by_name(&aos, "pol_angle").expect("aos aux should be present");
        let columnar_aux = NamedEventAccess::aux_by_name(&columnar, "pol_angle")
            .expect("columnar aux should be present");
        assert_relative_eq!(aos_aux, columnar_aux, epsilon = 1e-12);

        assert_relative_eq!(
            EventAccess::weight(&aos),
            EventAccess::weight(&columnar),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_dataset_filtering() {
        let metadata = Arc::new(
            DatasetMetadata::new(vec!["beam"], Vec::<String>::new())
                .expect("metadata should be valid"),
        );
        let events = vec![
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(0.0)],
                aux: vec![],
                weight: 1.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(0.5)],
                aux: vec![],
                weight: 1.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 5.0).with_mass(1.1)],
                // HACK: using 1.0 messes with this test because the eventual computation gives a mass
                // slightly less than 1.0
                aux: vec![],
                weight: 1.0,
            }),
        ];
        let dataset = Dataset::new_with_metadata(events, metadata);

        let metadata = dataset.metadata_arc();
        let mut mass = Mass::new(["beam"]);
        mass.bind(metadata.as_ref()).unwrap();
        let expression = mass.gt(0.0).and(&mass.lt(1.0));

        let filtered = dataset.filter(&expression).unwrap();
        assert_eq!(filtered.n_events(), 1);
        assert_relative_eq!(
            mass.value(&filtered.event(0).expect("event should exist")),
            0.5
        );
    }

    #[test]
    fn test_dataset_boost() {
        let dataset = test_dataset();
        let dataset_boosted = dataset.boost_to_rest_frame_of(&["proton", "kshort1", "kshort2"]);
        let p4_sum = dataset_boosted
            .event(0)
            .expect("event should exist")
            .get_p4_sum(["proton", "kshort1", "kshort2"]);
        assert_relative_eq!(p4_sum.px(), 0.0);
        assert_relative_eq!(p4_sum.py(), 0.0);
        assert_relative_eq!(p4_sum.pz(), 0.0, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn test_named_event_view() {
        let dataset = test_dataset();
        let view = dataset.named_event(0).expect("event should exist");
        let dataset_event = dataset.event(0).expect("event should exist");
        assert_relative_eq!(view.weight(), dataset_event.weight);
        let beam = view.p4("beam").expect("beam p4");
        assert_relative_eq!(beam.px(), dataset_event.p4s[0].px());
        assert_relative_eq!(beam.e(), dataset_event.p4s[0].e());

        let summed = view.get_p4_sum(["kshort1", "kshort2"]);
        assert_relative_eq!(
            summed.e(),
            dataset_event.p4s[2].e() + dataset_event.p4s[3].e()
        );

        let aux_angle = view.aux().get("pol_angle").copied().expect("pol angle");
        assert_relative_eq!(aux_angle, dataset_event.aux[1]);

        let metadata = dataset.metadata_arc();
        let boosted = view.boost_to_rest_frame_of(["proton", "kshort1", "kshort2"]);
        let boosted_event = Event::new(Arc::new(boosted), metadata);
        let boosted_sum = boosted_event.get_p4_sum(["proton", "kshort1", "kshort2"]);
        assert_relative_eq!(boosted_sum.px(), 0.0);
    }

    #[test]
    fn test_dataset_evaluate() {
        let dataset = test_dataset();
        let mass = Mass::new(["proton"]);
        assert_relative_eq!(dataset.evaluate(&mass).unwrap()[0], 1.007);
    }

    #[test]
    fn test_dataset_metadata_rejects_duplicate_names() {
        let err = DatasetMetadata::new(vec!["beam", "beam"], Vec::<String>::new());
        assert!(matches!(
            err,
            Err(LadduError::DuplicateName { category, .. }) if category == "p4"
        ));
        let err = DatasetMetadata::new(
            vec!["beam"],
            vec!["pol_angle".to_string(), "pol_angle".to_string()],
        );
        assert!(matches!(
            err,
            Err(LadduError::DuplicateName { category, .. }) if category == "aux"
        ));
    }

    #[test]
    fn test_dataset_lookup_by_name() {
        let dataset = test_dataset();
        let proton = dataset.p4_by_name(0, "proton").expect("proton p4");
        let proton_idx = dataset.metadata().p4_index("proton").unwrap();
        assert_relative_eq!(
            proton.e(),
            dataset.event(0).expect("event should exist").p4s[proton_idx].e()
        );
        assert!(dataset.p4_by_name(0, "unknown").is_none());
        let angle = dataset.aux_by_name(0, "pol_angle").expect("pol_angle");
        assert_relative_eq!(angle, dataset.event(0).expect("event should exist").aux[1]);
        assert!(dataset.aux_by_name(0, "missing").is_none());
    }

    #[test]
    fn test_binned_dataset() {
        let dataset = Dataset::new(vec![
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 1.0).with_mass(1.0)],
                aux: vec![],
                weight: 1.0,
            }),
            Arc::new(EventData {
                p4s: vec![Vec3::new(0.0, 0.0, 2.0).with_mass(2.0)],
                aux: vec![],
                weight: 2.0,
            }),
        ]);

        #[derive(Clone, Serialize, Deserialize, Debug)]
        struct BeamEnergy;
        impl Display for BeamEnergy {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "BeamEnergy")
            }
        }
        #[typetag::serde]
        impl Variable for BeamEnergy {
            fn value(&self, event: &EventData) -> f64 {
                event.p4s[0].e()
            }
        }
        assert_eq!(BeamEnergy.to_string(), "BeamEnergy");

        // Test binning by first particle energy
        let binned = dataset.bin_by(BeamEnergy, 2, (0.0, 3.0)).unwrap();

        assert_eq!(binned.n_bins(), 2);
        assert_eq!(binned.edges().len(), 3);
        assert_relative_eq!(binned.edges()[0], 0.0);
        assert_relative_eq!(binned.edges()[2], 3.0);
        assert_eq!(binned[0].n_events(), 1);
        assert_relative_eq!(binned[0].n_events_weighted(), 1.0);
        assert_eq!(binned[1].n_events(), 1);
        assert_relative_eq!(binned[1].n_events_weighted(), 2.0);
    }

    #[test]
    fn test_dataset_bootstrap() {
        let metadata = test_dataset().metadata_arc();
        let dataset = Dataset::new_with_metadata(
            vec![
                Arc::new(test_event()),
                Arc::new(EventData {
                    p4s: test_event().p4s.clone(),
                    aux: test_event().aux.clone(),
                    weight: 1.0,
                }),
            ],
            metadata,
        );
        assert_relative_ne!(
            dataset.event(0).expect("event should exist").weight,
            dataset.event(1).expect("event should exist").weight
        );

        let bootstrapped = dataset.bootstrap(43);
        assert_eq!(bootstrapped.n_events(), dataset.n_events());
        assert_relative_eq!(
            bootstrapped.event(0).expect("event should exist").weight,
            bootstrapped.event(1).expect("event should exist").weight
        );

        // Test empty dataset bootstrap
        let empty_dataset = Dataset::new(Vec::new());
        let empty_bootstrap = empty_dataset.bootstrap(43);
        assert_eq!(empty_bootstrap.n_events(), 0);
    }

    #[test]
    fn test_dataset_iteration_returns_events() {
        let dataset = test_dataset();
        let mut weights = Vec::new();
        for event in dataset.iter() {
            weights.push(event.weight());
        }
        assert_eq!(weights.len(), dataset.n_events());
        assert_relative_eq!(
            weights[0],
            dataset.event(0).expect("event should exist").weight
        );
    }

    #[test]
    fn test_dataset_into_iter_returns_events() {
        let dataset = test_dataset();
        let weights: Vec<f64> = dataset.into_iter().map(|event| event.weight()).collect();
        assert_eq!(weights.len(), 1);
        assert_relative_eq!(weights[0], test_event().weight);
    }

    #[test]
    fn test_dataset_get_event_local_reuses_underlying_data() {
        let dataset = test_dataset();
        let first = dataset.get_event(0).expect("event should exist");
        let second = dataset.get_event(0).expect("event should exist");
        assert!(Arc::ptr_eq(&first.data_arc(), &second.data_arc()));
    }

    #[test]
    fn test_dataset_event_out_of_bounds_is_error() {
        let dataset = test_dataset();
        assert!(dataset.event(99).is_err());
        assert!(dataset.get_event(99).is_none());
    }

    #[test]
    fn test_dataset_event_stress_local_repeated_access() {
        let metadata = test_dataset().metadata_arc();
        let base = test_event();
        let mut events = Vec::new();
        for idx in 0..8 {
            events.push(Arc::new(EventData {
                p4s: base.p4s.clone(),
                aux: base.aux.clone(),
                weight: 1.0 + idx as f64,
            }));
        }
        let dataset = Dataset::new_with_metadata(events, metadata);
        let baseline: Vec<f64> = (0..dataset.n_events())
            .map(|index| dataset.event(index).expect("event should exist").weight())
            .collect();

        for _ in 0..250 {
            for (index, expected_weight) in baseline.iter().enumerate() {
                let event = dataset.event(index).expect("event should exist");
                assert_relative_eq!(event.weight(), *expected_weight);
            }
        }
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn test_dataset_event_mpi_repeated_access_is_stable() {
        use_mpi(true);
        if get_world().is_none() {
            finalize_mpi();
            return;
        }

        let dataset = test_dataset();
        for _ in 0..32 {
            let first = dataset.event(0).expect("event should exist");
            let second = dataset.event(0).expect("event should exist");
            assert_relative_eq!(first.weight(), second.weight());
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn test_dataset_event_stress_mpi_repeated_access() {
        use_mpi(true);
        if get_world().is_none() {
            finalize_mpi();
            return;
        }

        let metadata = test_dataset().metadata_arc();
        let base = test_event();
        let mut events = Vec::new();
        for idx in 0..8 {
            events.push(Arc::new(EventData {
                p4s: base.p4s.clone(),
                aux: base.aux.clone(),
                weight: 1.0 + idx as f64,
            }));
        }
        let dataset = Dataset::new_with_metadata(events, metadata);

        let baseline: Vec<f64> = (0..dataset.n_events())
            .map(|index| dataset.event(index).expect("event should exist").weight())
            .collect();

        for _ in 0..120 {
            for (index, expected_weight) in baseline.iter().enumerate() {
                let event = dataset.event(index).expect("event should exist");
                assert_relative_eq!(event.weight(), *expected_weight);
            }
        }
        finalize_mpi();
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn test_dataset_iter_stress_mpi_repeated_passes() {
        use_mpi(true);
        if get_world().is_none() {
            finalize_mpi();
            return;
        }

        let metadata = test_dataset().metadata_arc();
        let base = test_event();
        let mut events = Vec::new();
        for idx in 0..8 {
            events.push(Arc::new(EventData {
                p4s: base.p4s.clone(),
                aux: base.aux.clone(),
                weight: 1.0 + idx as f64,
            }));
        }
        let dataset = Dataset::new_with_metadata(events, metadata);
        let baseline: Vec<f64> = dataset.iter().map(|event| event.weight()).collect();

        for _ in 0..80 {
            let current: Vec<f64> = dataset.iter().map(|event| event.weight()).collect();
            assert_eq!(current.len(), baseline.len());
            for (current_weight, expected_weight) in current.iter().zip(baseline.iter()) {
                assert_relative_eq!(*current_weight, *expected_weight);
            }
        }
        finalize_mpi();
    }

    #[test]
    fn test_event_display() {
        let event = test_event();
        let display_string = format!("{}", event);
        assert!(display_string.contains("Event:"));
        assert!(display_string.contains("p4s:"));
        assert!(display_string.contains("aux:"));
        assert!(display_string.contains("aux[0]: 0.38562805"));
        assert!(display_string.contains("aux[1]: 0.05708078"));
        assert!(display_string.contains("weight:"));
    }

    #[test]
    fn test_name_based_access() {
        let metadata =
            Arc::new(DatasetMetadata::new(vec!["beam", "target"], vec!["pol_angle"]).unwrap());
        let event = Arc::new(EventData {
            p4s: vec![Vec4::new(0.0, 0.0, 1.0, 1.0), Vec4::new(0.1, 0.2, 0.3, 0.5)],
            aux: vec![0.42],
            weight: 1.0,
        });
        let dataset = Dataset::new_with_metadata(vec![event], metadata);
        let beam = dataset.p4_by_name(0, "beam").unwrap();
        assert_relative_eq!(beam.px(), 0.0);
        assert_relative_eq!(beam.py(), 0.0);
        assert_relative_eq!(beam.pz(), 1.0);
        assert_relative_eq!(beam.e(), 1.0);
        assert_relative_eq!(dataset.aux_by_name(0, "pol_angle").unwrap(), 0.42);
        assert!(dataset.p4_by_name(0, "missing").is_none());
        assert!(dataset.aux_by_name(0, "missing").is_none());
    }

    #[test]
    fn test_parquet_roundtrip_to_tempfile() {
        let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
        let dir = make_temp_dir();
        let path = dir.join("roundtrip.parquet");
        let path_str = path.to_str().expect("path should be valid UTF-8");

        write_parquet(&dataset, path_str, &DatasetWriteOptions::default())
            .expect("writing parquet should succeed");
        let reopened = read_parquet(path_str, &DatasetReadOptions::new())
            .expect("parquet roundtrip should reopen");

        assert_datasets_close(&dataset, &reopened, TEST_P4_NAMES, TEST_AUX_NAMES);
        fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
    }

    #[test]
    fn test_read_parquet_columnar_matches_aos_to_columnar() {
        let path = test_data_path("data_f32.parquet");
        let path_str = path.to_str().expect("path should be valid UTF-8");
        let aos = read_parquet(path_str, &DatasetReadOptions::new()).expect("aos load should work");
        let columnar_from_aos = aos.to_storage().expect("aos->columnar should work");
        let columnar_direct = read_parquet_storage(path_str, &DatasetReadOptions::new())
            .expect("columnar load should work");
        assert_dataset_columnar_close(&columnar_from_aos, &columnar_direct);
    }

    #[test]
    fn test_parquet_columnar_roundtrip_to_tempfile() {
        let source_path = test_data_path("data_f32.parquet");
        let source_path_str = source_path.to_str().expect("path should be valid UTF-8");
        let dataset_columnar = read_parquet_storage(source_path_str, &DatasetReadOptions::new())
            .expect("columnar load");
        let dir = make_temp_dir();
        let path = dir.join("roundtrip_columnar.parquet");
        let path_str = path.to_str().expect("path should be valid UTF-8");

        write_parquet_storage(&dataset_columnar, path_str, &DatasetWriteOptions::default())
            .expect("writing columnar parquet should succeed");
        let reopened = read_parquet_storage(path_str, &DatasetReadOptions::new())
            .expect("columnar roundtrip reopen");

        assert_dataset_columnar_close(&dataset_columnar, &reopened);
        fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
    }

    #[test]
    fn test_read_root_columnar_matches_root_aos_to_columnar() {
        let path = test_data_path("data_f32.root");
        let path_str = path.to_str().expect("path should be valid UTF-8");
        let aos =
            read_root(path_str, &DatasetReadOptions::new()).expect("root aos load should work");
        let columnar_from_aos = aos.to_storage().expect("aos->columnar should work");
        let columnar_direct = read_root_storage(path_str, &DatasetReadOptions::new())
            .expect("root columnar load should work");
        assert_dataset_columnar_close(&columnar_from_aos, &columnar_direct);
    }

    #[test]
    fn test_read_root_columnar_matches_parquet_columnar() {
        let root_path = test_data_path("data_f32.root");
        let root_path_str = root_path.to_str().expect("path should be valid UTF-8");
        let parquet_path = test_data_path("data_f32.parquet");
        let parquet_path_str = parquet_path.to_str().expect("path should be valid UTF-8");

        let from_root = read_root_storage(root_path_str, &DatasetReadOptions::new())
            .expect("root columnar load should work");
        let from_parquet = read_parquet_storage(parquet_path_str, &DatasetReadOptions::new())
            .expect("parquet columnar load should work");
        assert_dataset_columnar_close(&from_root, &from_parquet);
    }

    #[test]
    fn test_root_roundtrip_to_tempfile() {
        let dataset = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
        let dir = make_temp_dir();
        let path = dir.join("roundtrip.root");
        let path_str = path.to_str().expect("path should be valid UTF-8");

        write_root(&dataset, path_str, &DatasetWriteOptions::default())
            .expect("writing root should succeed");
        let reopened =
            read_root(path_str, &DatasetReadOptions::new()).expect("root roundtrip should reopen");

        assert_datasets_close(&dataset, &reopened, TEST_P4_NAMES, TEST_AUX_NAMES);
        fs::remove_dir_all(&dir).expect("temp dir cleanup should succeed");
    }
}
