use accurate::{sum::Klein, traits::*};
use arrow::{
    array::{Float32Array, Float64Array},
    datatypes::DataType,
    record_batch::RecordBatch,
};
use auto_ops::impl_op_ex;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::path::Path;
use std::{fmt::Display, fs::File};
use std::{path::PathBuf, sync::Arc};

use oxyroot::{Branch, Named, ReaderTree, RootFile};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;

use crate::utils::get_bin_edges;
use crate::{
    utils::{
        variables::{Variable, VariableExpression},
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
    EventData {
        p4s: vec![
            Vec3::new(0.0, 0.0, 8.747).with_mass(0.0),         // beam
            Vec3::new(0.119, 0.374, 0.222).with_mass(1.007),   // "proton"
            Vec3::new(-0.112, 0.293, 3.081).with_mass(0.498),  // "kaon"
            Vec3::new(-0.007, -0.667, 5.446).with_mass(0.498), // "kaon"
        ],
        aux: vec![0.38562805, 1.93592989],
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

/// A collection of [`EventData`].
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    pub(crate) p4_names: Vec<String>,
    pub(crate) aux_names: Vec<String>,
    pub(crate) p4_lookup: IndexMap<String, usize>,
    pub(crate) aux_lookup: IndexMap<String, usize>,
}

impl DatasetMetadata {
    /// Construct metadata from explicit particle and auxiliary names.
    pub fn new<P: Into<String>, A: Into<String>>(
        p4_names: Vec<P>,
        aux_names: Vec<A>,
    ) -> LadduResult<Self> {
        let mut p4_lookup = IndexMap::with_capacity(p4_names.len());
        let mut aux_lookup = IndexMap::with_capacity(aux_names.len());
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
        })
    }

    /// Create metadata with no registered names.
    pub fn empty() -> Self {
        Self {
            p4_names: Vec::new(),
            aux_names: Vec::new(),
            p4_lookup: IndexMap::new(),
            aux_lookup: IndexMap::new(),
        }
    }

    /// Resolve the index of a four-momentum by name.
    pub fn p4_index(&self, name: &str) -> Option<usize> {
        self.p4_lookup.get(name).copied()
    }

    /// Resolve the index of an auxiliary scalar by name.
    pub fn aux_index(&self, name: &str) -> Option<usize> {
        self.aux_lookup.get(name).copied()
    }
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self::empty()
    }
}

/// A collection of [`EventData`] with optional metadata for name-based lookups.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// The [`EventData`] contained in the [`Dataset`]
    pub events: Vec<Event>,
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

    /// Retrieve a four-momentum by name.
    pub fn p4(&self, name: &str) -> Option<&Vec4> {
        self.metadata
            .p4_index(name)
            .and_then(|idx| self.event.p4s.get(idx))
    }

    fn resolve_p4_indices<N>(&self, names: N) -> Vec<usize>
    where
        N: IntoIterator,
        N::Item: AsRef<str>,
    {
        names
            .into_iter()
            .map(|name| {
                let name_ref = name.as_ref();
                self.metadata
                    .p4_index(name_ref)
                    .unwrap_or_else(|| panic!("Unknown particle name '{name}'", name = name_ref))
            })
            .collect()
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

impl Dataset {
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
    pub fn named_event(&self, index: usize) -> Event {
        self.events[index].clone()
    }

    /// Retrieve a four-momentum by name for the event at `event_index`.
    pub fn p4_by_name(&self, event_index: usize, name: &str) -> Option<Vec4> {
        let idx = self.p4_index(name)?;
        self.events
            .get(event_index)
            .and_then(|event| event.p4s.get(idx))
            .copied()
    }

    /// Retrieve an auxiliary scalar by name for the event at `event_index`.
    pub fn aux_by_name(&self, event_index: usize, name: &str) -> Option<f64> {
        let idx = self.aux_index(name)?;
        self.events
            .get(event_index)
            .and_then(|event| event.aux.get(idx))
            .copied()
    }

    /// Get a reference to the [`EventData`] at the given index in the [`Dataset`] (non-MPI
    /// version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just index into a [`Dataset`]
    /// as if it were any other [`Vec`]:
    ///
    /// ```ignore
    /// let ds: Dataset = Dataset::new(events);
    /// let event_0 = ds[0];
    /// ```
    pub fn index_local(&self, index: usize) -> &Event {
        &self.events[index]
    }

    #[cfg(feature = "mpi")]
    fn partition(
        events: Vec<Arc<EventData>>,
        world: &SimpleCommunicator,
    ) -> Vec<Vec<Arc<EventData>>> {
        let (counts, displs) = world.get_counts_displs(events.len());
        counts
            .iter()
            .zip(displs.iter())
            .map(|(&count, &displ)| {
                events
                    .iter()
                    .skip(displ as usize)
                    .take(count as usize)
                    .cloned()
                    .collect()
            })
            .collect()
    }

    /// Get a reference to the [`EventData`] at the given index in the [`Dataset`]
    /// (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just index into a [`Dataset`]
    /// as if it were any other [`Vec`]:
    ///
    /// ```ignore
    /// let ds: Dataset = Dataset::new(events);
    /// let event_0 = ds[0];
    /// ```
    #[cfg(feature = "mpi")]
    pub fn index_mpi(&self, index: usize, world: &SimpleCommunicator) -> &Event {
        let total = self.n_events();
        let (owning_rank, local_index) = world.owner_of_global_index(index, total);
        let mut serialized_event_buffer_len: usize = 0;
        let mut serialized_event_buffer: Vec<u8> = Vec::default();
        let config = bincode::config::standard();
        if world.rank() == owning_rank {
            let event = self.index_local(local_index);
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
        let (event, _): (EventData, usize) =
            bincode::serde::decode_from_slice(&serialized_event_buffer[..], config).unwrap();
        let leaked = Event::new(Arc::new(event), self.metadata.clone());
        Box::leak(Box::new(leaked))
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = &'a Event;
    type IntoIter = std::slice::Iter<'a, Event>;

    fn into_iter(self) -> Self::IntoIter {
        self.events.iter()
    }
}

impl IntoIterator for Dataset {
    type Item = Event;
    type IntoIter = std::vec::IntoIter<Event>;

    fn into_iter(self) -> Self::IntoIter {
        self.events.into_iter()
    }
}

impl Index<usize> for Dataset {
    type Output = Event;

    fn index(&self, index: usize) -> &Self::Output {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.index_mpi(index, &world);
            }
        }
        self.index_local(index)
    }
}

impl Dataset {
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
            .collect();
        Dataset {
            events: wrapped_events,
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
        let local = partitions[world.rank() as usize]
            .iter()
            .cloned()
            .map(|event| Event::new(event, metadata.clone()))
            .collect();
        Dataset {
            events: local,
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

    /// The number of [`EventData`]s in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events`] instead.
    pub fn n_events_local(&self) -> usize {
        self.events.len()
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
        #[cfg(feature = "rayon")]
        return self.events.par_iter().map(|e| e.weight).collect();
        #[cfg(not(feature = "rayon"))]
        return self.events.iter().map(|e| e.weight).collect();
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
            .events
            .par_iter()
            .map(|e| e.weight)
            .parallel_sum_with_accumulator::<Klein<f64>>();
        #[cfg(not(feature = "rayon"))]
        return self.events.iter().map(|e| e.weight).sum();
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
        let indices: Vec<usize> = names
            .iter()
            .map(|name| {
                let name_ref = name.as_ref();
                self.metadata
                    .p4_index(name_ref)
                    .unwrap_or_else(|| panic!("Unknown particle name '{name}'", name = name_ref))
            })
            .collect();
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

    /// Open a file and construct a [`Dataset`] using named columns.
    ///
    /// Each entry in `p4_names` identifies a particle; the loader expects columns named
    /// ``{identifier}_px``, ``{identifier}_py``, ``{identifier}_pz``, and ``{identifier}_e`` for
    /// each identifier (component suffixes are matched case-insensitively). Auxiliary scalars are
    /// listed explicitly via `aux_names` and stored in the same order. Columns may be encoded as
    /// `Float32` or `Float64` and are promoted to `f64` on load. If `boost_to_restframe_of` is
    /// provided, the method boosts every event to the rest frame of the specified particles.
    ///
    /// This method currently supports Parquet files and ROOT TTrees. When reading
    /// ROOT files containing multiple trees, set [`DatasetReadOptions::tree`] to select one.
    pub fn open(file_path: &str, options: &DatasetReadOptions) -> LadduResult<Arc<Dataset>> {
        let path = Path::new(&*shellexpand::full(file_path)?).canonicalize()?;
        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                match ext_str.to_lowercase().as_str() {
                    "parquet" => Self::open_parquet_impl(path, options),
                    "root" => Self::open_root_impl(path, options),
                    _ => Err(LadduError::Custom(format!(
                        "Filetype not supported: {}",
                        ext_str
                    ))),
                }
            } else {
                Err(LadduError::Custom(format!(
                    "Invalid file extension: {}",
                    path.display()
                )))
            }
        } else {
            Err(LadduError::Custom(format!(
                "Filetype could not be detected from extension: {}",
                path.display()
            )))
        }
    }
    fn open_parquet_impl(
        file_path: PathBuf,
        options: &DatasetReadOptions,
    ) -> LadduResult<Arc<Dataset>> {
        let (detected_p4_names, detected_aux_names) = detect_columns(&file_path)?;
        let (metadata, boost_indices) =
            options.resolve_metadata(detected_p4_names, detected_aux_names)?;
        let file = File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;
        let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

        #[cfg(feature = "rayon")]
        let events: Vec<Arc<EventData>> = {
            let boost_indices = boost_indices.as_deref();
            let batch_events: Vec<LadduResult<Vec<Arc<EventData>>>> = batches
                .into_par_iter()
                .map(|batch| {
                    record_batch_to_events(
                        batch,
                        &metadata.p4_names,
                        &metadata.aux_names,
                        boost_indices,
                    )
                })
                .collect();
            let mut events = Vec::new();
            for batch in batch_events {
                let mut batch = batch?;
                events.append(&mut batch);
            }
            events
        };

        #[cfg(not(feature = "rayon"))]
        let events: Vec<Arc<EventData>> = {
            let boost_indices = boost_indices.as_deref();
            batches
                .into_iter()
                .map(|batch| {
                    record_batch_to_events(
                        batch,
                        &metadata.p4_names,
                        &metadata.aux_names,
                        boost_indices,
                    )
                })
                .collect::<LadduResult<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect()
        };

        Ok(Arc::new(Dataset::new_with_metadata(events, metadata)))
    }

    fn open_root_impl(
        file_path: PathBuf,
        options: &DatasetReadOptions,
    ) -> LadduResult<Arc<Dataset>> {
        let mut file = RootFile::open(&file_path).map_err(|err| {
            LadduError::Custom(format!(
                "Failed to open ROOT file '{}': {err}",
                file_path.display()
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
        let (metadata, boost_indices) =
            options.resolve_metadata(detected_p4_names, detected_aux_names)?;

        let weight_values = read_branch_values(&lookup, "weight")?;
        let n_events = weight_values.len();

        struct RootP4Columns {
            px: Vec<f64>,
            py: Vec<f64>,
            pz: Vec<f64>,
            e: Vec<f64>,
        }

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

        let boost_indices_slice = boost_indices.as_deref();

        let mut events = Vec::with_capacity(n_events);
        for row in 0..n_events {
            let mut p4s = Vec::with_capacity(p4_columns.len());
            for columns in &p4_columns {
                p4s.push(Vec4::new(
                    columns.px[row],
                    columns.py[row],
                    columns.pz[row],
                    columns.e[row],
                ));
            }

            let mut aux = Vec::with_capacity(aux_columns.len());
            for column in &aux_columns {
                aux.push(column[row]);
            }

            let mut event = EventData {
                p4s,
                aux,
                weight: weight_values[row],
            };
            if let Some(indices) = boost_indices_slice {
                if !indices.is_empty() {
                    event = event.boost_to_rest_frame_of(indices);
                }
            }
            events.push(Arc::new(event));
        }

        Ok(Arc::new(Dataset::new_with_metadata(events, metadata)))
    }
}

impl_op_ex!(+ |a: &Dataset, b: &Dataset| -> Dataset {
    debug_assert_eq!(a.metadata.p4_names, b.metadata.p4_names);
    debug_assert_eq!(a.metadata.aux_names, b.metadata.aux_names);
    Dataset {
        events: a
            .events
            .iter()
            .chain(b.events.iter())
            .cloned()
            .collect(),
        metadata: a.metadata.clone(),
    }
});

/// Options for reading a [`Dataset`] from a file.
///
/// # See Also
/// [`Dataset::open`]
#[derive(Default, Clone)]
pub struct DatasetReadOptions {
    /// Particle names to read from the data file.
    pub p4_names: Option<Vec<String>>,
    /// Auxiliary scalar names to read from the data file.
    pub aux_names: Option<Vec<String>>,
    /// Particles whose rest frame should be used to boost each event.
    pub boost_to_restframe_of: Option<Vec<String>>,
    /// Name of the tree to read when loading ROOT files. When absent and the file contains a
    /// single tree, it will be selected automatically.
    pub tree: Option<String>,
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

    /// If provided, [`Event`]s in the [`Dataset`] will be boosted to the rest frame of the specified particles.
    pub fn boost_to_restframe_of<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.boost_to_restframe_of =
            Some(names.into_iter().map(|s| s.as_ref().to_string()).collect());
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

    fn resolve_metadata(
        &self,
        detected_p4_names: Vec<String>,
        detected_aux_names: Vec<String>,
    ) -> LadduResult<(Arc<DatasetMetadata>, Option<Vec<usize>>)> {
        let p4_names_vec = self.p4_names.clone().unwrap_or(detected_p4_names);
        let aux_names_vec = self.aux_names.clone().unwrap_or(detected_aux_names);

        let metadata = Arc::new(DatasetMetadata::new(p4_names_vec, aux_names_vec)?);

        let boost_indices = if let Some(names) = self.boost_to_restframe_of.clone() {
            let mut indices = Vec::with_capacity(names.len());
            for name in names {
                let idx = metadata
                    .p4_index(&name)
                    .ok_or_else(|| LadduError::UnknownName {
                        category: "p4",
                        name: name.to_string(),
                    })?;
                indices.push(idx);
            }
            Some(indices)
        } else {
            None
        };

        Ok((metadata, boost_indices))
    }
}

const P4_COMPONENT_SUFFIXES: [&str; 4] = ["_px", "_py", "_pz", "_e"];

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

fn prepare_float_column_from_candidates<'a>(
    batch: &'a RecordBatch,
    candidates: &[String],
    logical_name: &str,
) -> LadduResult<FloatColumn<'a>> {
    use arrow::datatypes::DataType;

    for candidate in candidates {
        if let Some(column) = batch.column_by_name(candidate) {
            return match column.data_type() {
                DataType::Float32 => Ok(FloatColumn::F32(
                    column
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("Column advertised as Float32 but could not be downcast"),
                )),
                DataType::Float64 => Ok(FloatColumn::F64(
                    column
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .expect("Column advertised as Float64 but could not be downcast"),
                )),
                other => {
                    return Err(LadduError::InvalidColumnType {
                        name: candidate.clone(),
                        datatype: other.to_string(),
                    })
                }
            };
        }
    }
    Err(LadduError::MissingColumn {
        name: logical_name.to_string(),
    })
}

fn record_batch_to_events(
    batch: RecordBatch,
    p4_names: &[String],
    aux_names: &[String],
    boost_indices: Option<&[usize]>,
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

    let weight_column = prepare_float_column(batch_ref, "weight")?;

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

        let mut event = EventData {
            p4s,
            aux,
            weight: weight_column.value(row),
        };
        if let Some(indices) = boost_indices {
            if !indices.is_empty() {
                event = event.boost_to_rest_frame_of(indices);
            }
        }
        events.push(Arc::new(event));
    }
    Ok(events)
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
    use crate::utils::vectors::Vec3;
    use approx::{assert_relative_eq, assert_relative_ne};
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};

    fn test_data_path(file: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join(file)
    }

    fn open_test_dataset(file: &str, options: DatasetReadOptions) -> Arc<Dataset> {
        let path = test_data_path(file);
        let path_str = path.to_str().expect("test data path should be valid UTF-8");
        Dataset::open(path_str, &options).expect("dataset should open")
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
            let levent = &left[idx];
            let revent = &right[idx];
            assert_events_close(levent, revent, p4_names, aux_names);
        }
    }

    #[test]
    fn test_open_parquet_auto_matches_explicit_names() {
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
    fn test_open_parquet_f64_matches_f32() {
        let f32_ds = open_test_dataset("data_f32.parquet", DatasetReadOptions::new());
        let f64_ds = open_test_dataset("data_f64.parquet", DatasetReadOptions::new());
        assert_datasets_close(&f64_ds, &f32_ds, TEST_P4_NAMES, TEST_AUX_NAMES);
    }

    #[test]
    fn test_open_root_detects_columns_and_matches_parquet() {
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
    fn test_open_root_f64_matches_parquet() {
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
        assert_eq!(dataset_sum[0].weight, dataset[0].weight);
        assert_eq!(dataset_sum[1].weight, dataset2[0].weight);
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
        assert_relative_eq!(mass.value(&filtered[0]), 0.5);
    }

    #[test]
    fn test_dataset_boost() {
        let dataset = test_dataset();
        let dataset_boosted = dataset.boost_to_rest_frame_of(&["proton", "kshort1", "kshort2"]);
        let p4_sum = dataset_boosted[0].get_p4_sum(["proton", "kshort1", "kshort2"]);
        assert_relative_eq!(p4_sum.px(), 0.0);
        assert_relative_eq!(p4_sum.py(), 0.0);
        assert_relative_eq!(p4_sum.pz(), 0.0, epsilon = f64::EPSILON.sqrt());
    }

    #[test]
    fn test_named_event_view() {
        let dataset = test_dataset();
        let view = dataset.named_event(0);

        assert_relative_eq!(view.weight(), dataset[0].weight);
        let beam = view.p4("beam").expect("beam p4");
        assert_relative_eq!(beam.px(), dataset[0].p4s[0].px());
        assert_relative_eq!(beam.e(), dataset[0].p4s[0].e());

        let summed = view.get_p4_sum(["kshort1", "kshort2"]);
        assert_relative_eq!(summed.e(), dataset[0].p4s[2].e() + dataset[0].p4s[3].e());

        let aux_angle = view.aux().get("pol_angle").copied().expect("pol angle");
        assert_relative_eq!(aux_angle, dataset[0].aux[1]);

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
        assert_relative_ne!(dataset[0].weight, dataset[1].weight);

        let bootstrapped = dataset.bootstrap(43);
        assert_eq!(bootstrapped.n_events(), dataset.n_events());
        assert_relative_eq!(bootstrapped[0].weight, bootstrapped[1].weight);

        // Test empty dataset bootstrap
        let empty_dataset = Dataset::new(Vec::new());
        let empty_bootstrap = empty_dataset.bootstrap(43);
        assert_eq!(empty_bootstrap.n_events(), 0);
    }

    #[test]
    fn test_dataset_iteration_returns_events() {
        let dataset = test_dataset();
        let mut weights = Vec::new();
        for event in &dataset {
            weights.push(event.weight());
        }
        assert_eq!(weights.len(), dataset.n_events());
        assert_relative_eq!(weights[0], dataset[0].weight);
    }
    #[test]
    fn test_event_display() {
        let event = test_event();
        let display_string = format!("{}", event);
        assert!(display_string.contains("Event:"));
        assert!(display_string.contains("p4s:"));
        assert!(display_string.contains("aux:"));
        assert!(display_string.contains("aux[0]: 0.38562805"));
        assert!(display_string.contains("aux[1]: 1.93592989"));
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
}
