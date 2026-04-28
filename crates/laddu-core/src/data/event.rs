use std::{fmt::Display, ops::Deref, sync::Arc};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::{Dataset, DatasetMetadata};
use crate::{
    variables::Variable,
    vectors::{Vec3, Vec4},
};

/// An event that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitudes::Amplitude). This particular event contains the reaction
/// $`\gamma p \to K_S^0 K_S^0 p`$ with a polarized photon beam.
pub fn test_event() -> EventData {
    let pol_magnitude = 0.38562805;
    let pol_angle = 0.05708078;
    EventData {
        p4s: vec![
            Vec3::new(0.0, 0.0, 8.747).with_mass(0.0),
            Vec3::new(0.119, 0.374, 0.222).with_mass(1.007),
            Vec3::new(-0.112, 0.293, 3.081).with_mass(0.498),
            Vec3::new(-0.007, -0.667, 5.446).with_mass(0.498),
        ],
        aux: vec![pol_magnitude, pol_angle],
        weight: 0.48,
    }
}

/// Particle names used by [`crate::data::test_dataset`].
pub const TEST_P4_NAMES: &[&str] = &["beam", "proton", "kshort1", "kshort2"];
/// Auxiliary scalar names used by [`crate::data::test_dataset`].
pub const TEST_AUX_NAMES: &[&str] = &["pol_magnitude", "pol_angle"];

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
}

#[derive(Debug, Clone, Default)]
/// Columnar storage for one named four-momentum across all events.
pub struct ColumnarP4Column {
    pub(crate) px: Vec<f64>,
    pub(crate) py: Vec<f64>,
    pub(crate) pz: Vec<f64>,
    pub(crate) e: Vec<f64>,
}

impl ColumnarP4Column {
    /// Create an empty four-momentum column with capacity for `capacity` events.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            px: Vec::with_capacity(capacity),
            py: Vec::with_capacity(capacity),
            pz: Vec::with_capacity(capacity),
            e: Vec::with_capacity(capacity),
        }
    }

    /// Append one four-momentum to the column.
    pub fn push(&mut self, p4: Vec4) {
        self.px.push(p4.x);
        self.py.push(p4.y);
        self.pz.push(p4.z);
        self.e.push(p4.t);
    }

    /// Return the four-momentum at `event_index`.
    pub fn get(&self, event_index: usize) -> Vec4 {
        Vec4::new(
            self.px[event_index],
            self.py[event_index],
            self.pz[event_index],
            self.e[event_index],
        )
    }
}

/// Columnar dataset storage used by [`Dataset`].
#[derive(Debug, Default)]
pub struct DatasetStorage {
    pub(crate) metadata: Arc<DatasetMetadata>,
    pub(crate) p4: Vec<ColumnarP4Column>,
    pub(crate) aux: Vec<Vec<f64>>,
    pub(crate) weights: Vec<f64>,
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
    /// Create columnar dataset storage from metadata, four-momentum columns, auxiliary columns, and weights.
    pub fn new(
        metadata: DatasetMetadata,
        p4: Vec<ColumnarP4Column>,
        aux: Vec<Vec<f64>>,
        weights: Vec<f64>,
    ) -> Self {
        Self {
            metadata: Arc::new(metadata),
            p4,
            aux,
            weights,
        }
    }
    /// Convert this columnar dataset back to a row-event dataset.
    pub fn to_dataset(&self) -> Dataset {
        let events = (0..self.n_events())
            .map(|event_index| Arc::new(self.event_data(event_index)))
            .collect::<Vec<_>>();
        #[cfg(not(feature = "mpi"))]
        let dataset = Dataset::new_local(events, self.metadata.clone());
        #[cfg(feature = "mpi")]
        let mut dataset = Dataset::new_local(events, self.metadata.clone());
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                dataset.set_cached_global_event_count_from_world(&world);
                dataset.set_cached_global_weighted_sum_from_world(&world);
            }
        }
        dataset
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
    pub(crate) fn p4(&self, event_index: usize, p4_index: usize) -> Vec4 {
        self.p4[p4_index].get(event_index)
    }

    /// Retrieve an aux value by row and aux index.
    pub(crate) fn aux(&self, event_index: usize, aux_index: usize) -> f64 {
        self.aux[aux_index][event_index]
    }

    /// Retrieve event weight by row index.
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
}

/// A name-aware columnar event view over a single row in a dataset.
#[derive(Debug)]
pub struct NamedEventView<'a> {
    row: ColumnarEventView<'a>,
    metadata: &'a DatasetMetadata,
}

impl NamedEventView<'_> {
    /// Retrieve a four-momentum by positional index.
    pub fn p4_at(&self, p4_index: usize) -> Vec4 {
        self.row.p4(p4_index)
    }

    /// Retrieve an auxiliary scalar by positional index.
    pub fn aux_at(&self, aux_index: usize) -> f64 {
        self.row.aux(aux_index)
    }

    /// Number of four-momenta in this event.
    pub fn n_p4(&self) -> usize {
        self.row.storage.p4.len()
    }

    /// Number of auxiliary values in this event.
    pub fn n_aux(&self) -> usize {
        self.row.storage.aux.len()
    }

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
        variable.value(self)
    }
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
