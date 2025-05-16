use accurate::{sum::Klein, traits::*};
use arrow::array::Float32Array;
use arrow::record_batch::RecordBatch;
use auto_ops::impl_op_ex;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::path::Path;
use std::sync::Arc;
use std::{fmt::Display, fs::File};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(feature = "mpi")]
use mpi::{datatype::PartitionMut, topology::SimpleCommunicator, traits::*};

#[cfg(feature = "mpi")]
use crate::mpi::LadduMPI;

use crate::utils::get_bin_edges;
use crate::{
    utils::{
        variables::Variable,
        vectors::{Vec3, Vec4},
    },
    Float, LadduError,
};

const P4_PREFIX: &str = "p4_";
const AUX_PREFIX: &str = "aux_";

/// An event that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitudes::Amplitude). This particular event contains the reaction
/// $`\gamma p \to K_S^0 K_S^0 p`$ with a polarized photon beam.
pub fn test_event() -> Event {
    use crate::utils::vectors::*;
    Event {
        p4s: vec![
            Vec3::new(0.0, 0.0, 8.747).with_mass(0.0),         // beam
            Vec3::new(0.119, 0.374, 0.222).with_mass(1.007),   // "proton"
            Vec3::new(-0.112, 0.293, 3.081).with_mass(0.498),  // "kaon"
            Vec3::new(-0.007, -0.667, 5.446).with_mass(0.498), // "kaon"
        ],
        aux: vec![Vec3::new(0.385, 0.022, 0.000)],
        weight: 0.48,
    }
}

/// An dataset that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitudes::Amplitude). This particular dataset contains a singular
/// [`Event`] generated from [`test_event`].
pub fn test_dataset() -> Dataset {
    Dataset::new(vec![Arc::new(test_event())])
}

/// A single event in a [`Dataset`] containing all the relevant particle information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Event {
    /// A list of four-momenta for each particle.
    pub p4s: Vec<Vec4>,
    /// A list of auxiliary vectors which can be used to store data like particle polarization.
    pub aux: Vec<Vec3>,
    /// The weight given to the event.
    pub weight: Float,
}

impl Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Event:")?;
        writeln!(f, "  p4s:")?;
        for p4 in &self.p4s {
            writeln!(f, "    {}", p4.to_p4_string())?;
        }
        writeln!(f, "  eps:")?;
        for eps_vec in &self.aux {
            writeln!(f, "    [{}, {}, {}]", eps_vec.x, eps_vec.y, eps_vec.z)?;
        }
        writeln!(f, "  weight:")?;
        writeln!(f, "    {}", self.weight)?;
        Ok(())
    }
}

impl Event {
    /// Return a four-momentum from the sum of four-momenta at the given indices in the [`Event`].
    pub fn get_p4_sum<T: AsRef<[usize]>>(&self, indices: T) -> Vec4 {
        indices.as_ref().iter().map(|i| self.p4s[*i]).sum::<Vec4>()
    }
    /// Boost all the four-momenta in the [`Event`] to the rest frame of the given set of
    /// four-momenta by indices.
    pub fn boost_to_rest_frame_of<T: AsRef<[usize]>>(&self, indices: T) -> Self {
        let frame = self.get_p4_sum(indices);
        Event {
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

/// A collection of [`Event`]s.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// The [`Event`]s contained in the [`Dataset`]
    pub events: Vec<Arc<Event>>,
}

impl Dataset {
    /// Get a reference to the [`Event`] at the given index in the [`Dataset`] (non-MPI
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
    fn get_rank_index(index: usize, displs: &[i32], world: &SimpleCommunicator) -> (i32, usize) {
        for (i, &displ) in displs.iter().enumerate() {
            if displ as usize > index {
                return (i as i32 - 1, index - displs[i - 1] as usize);
            }
        }
        (
            world.size() - 1,
            index - displs[world.size() as usize - 1] as usize,
        )
    }

    #[cfg(feature = "mpi")]
    fn partition(events: Vec<Arc<Event>>, world: &SimpleCommunicator) -> Vec<Vec<Arc<Event>>> {
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

    /// Get a reference to the [`Event`] at the given index in the [`Dataset`]
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
        let (_, displs) = world.get_counts_displs(self.n_events());
        let (owning_rank, local_index) = Dataset::get_rank_index(index, &displs, world);
        let mut serialized_event_buffer_len: usize = 0;
        let mut serialized_event_buffer: Vec<u8> = Vec::default();
        let config = bincode::config::standard();
        if world.rank() == owning_rank {
            let event = self.index_local(local_index);
            serialized_event_buffer = bincode::serde::encode_to_vec(event, config).unwrap();
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
        let (event, _): (Event, usize) =
            bincode::serde::decode_from_slice(&serialized_event_buffer[..], config).unwrap();
        Box::leak(Box::new(event))
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
    /// Create a new [`Dataset`] from a list of [`Event`]s (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::new`] instead.
    pub fn new_local(events: Vec<Arc<Event>>) -> Self {
        Dataset { events }
    }

    /// Create a new [`Dataset`] from a list of [`Event`]s (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::new`] instead.
    #[cfg(feature = "mpi")]
    pub fn new_mpi(events: Vec<Arc<Event>>, world: &SimpleCommunicator) -> Self {
        Dataset {
            events: Dataset::partition(events, world)[world.rank() as usize].clone(),
        }
    }

    /// Create a new [`Dataset`] from a list of [`Event`]s.
    ///
    /// This method is prefered for external use because it contains proper MPI construction
    /// methods. Constructing a [`Dataset`] manually is possible, but may cause issues when
    /// interfacing with MPI and should be avoided unless you know what you are doing.
    pub fn new(events: Vec<Arc<Event>>) -> Self {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return Dataset::new_mpi(events, &world);
            }
        }
        Dataset::new_local(events)
    }

    /// The number of [`Event`]s in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events`] instead.
    pub fn n_events_local(&self) -> usize {
        self.events.len()
    }

    /// The number of [`Event`]s in the [`Dataset`] (MPI-compatible version).
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

    /// The number of [`Event`]s in the [`Dataset`].
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
    /// Extract a list of weights over each [`Event`] in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::weights`] instead.
    pub fn weights_local(&self) -> Vec<Float> {
        #[cfg(feature = "rayon")]
        return self.events.par_iter().map(|e| e.weight).collect();
        #[cfg(not(feature = "rayon"))]
        return self.events.iter().map(|e| e.weight).collect();
    }

    /// Extract a list of weights over each [`Event`] in the [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::weights`] instead.
    #[cfg(feature = "mpi")]
    pub fn weights_mpi(&self, world: &SimpleCommunicator) -> Vec<Float> {
        let local_weights = self.weights_local();
        let n_events = self.n_events();
        let mut buffer: Vec<Float> = vec![0.0; n_events];
        let (counts, displs) = world.get_counts_displs(n_events);
        {
            let mut partitioned_buffer = PartitionMut::new(&mut buffer, counts, displs);
            world.all_gather_varcount_into(&local_weights, &mut partitioned_buffer);
        }
        buffer
    }

    /// Extract a list of weights over each [`Event`] in the [`Dataset`].
    pub fn weights(&self) -> Vec<Float> {
        #[cfg(feature = "mpi")]
        {
            if let Some(world) = crate::mpi::get_world() {
                return self.weights_mpi(&world);
            }
        }
        self.weights_local()
    }

    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`] (non-MPI version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events_weighted`] instead.
    pub fn n_events_weighted_local(&self) -> Float {
        #[cfg(feature = "rayon")]
        return self
            .events
            .par_iter()
            .map(|e| e.weight)
            .parallel_sum_with_accumulator::<Klein<Float>>();
        #[cfg(not(feature = "rayon"))]
        return self.events.iter().map(|e| e.weight).sum();
    }
    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`] (MPI-compatible version).
    ///
    /// # Notes
    ///
    /// This method is not intended to be called in analyses but rather in writing methods
    /// that have `mpi`-feature-gated versions. Most users should just call [`Dataset::n_events_weighted`] instead.
    #[cfg(feature = "mpi")]
    pub fn n_events_weighted_mpi(&self, world: &SimpleCommunicator) -> Float {
        let mut n_events_weighted_partitioned: Vec<Float> = vec![0.0; world.size() as usize];
        let n_events_weighted_local = self.n_events_weighted_local();
        world.all_gather_into(&n_events_weighted_local, &mut n_events_weighted_partitioned);
        #[cfg(feature = "rayon")]
        return n_events_weighted_partitioned
            .into_par_iter()
            .parallel_sum_with_accumulator::<Klein<Float>>();
        #[cfg(not(feature = "rayon"))]
        return n_events_weighted_partitioned.iter().sum();
    }

    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`].
    pub fn n_events_weighted(&self) -> Float {
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
        let bootstrapped_events: Vec<Arc<Event>> = indices
            .into_par_iter()
            .map(|idx| self.events[idx].clone())
            .collect();
        #[cfg(not(feature = "rayon"))]
        let bootstrapped_events: Vec<Arc<Event>> = indices
            .into_iter()
            .map(|idx| self.events[idx].clone())
            .collect();
        Arc::new(Dataset {
            events: bootstrapped_events,
        })
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
        #[cfg(feature = "rayon")]
        let bootstrapped_events: Vec<Arc<Event>> = indices
            .into_par_iter()
            .map(|idx| self.events[idx].clone())
            .collect();
        #[cfg(not(feature = "rayon"))]
        let bootstrapped_events: Vec<Arc<Event>> = indices
            .into_iter()
            .map(|idx| self.events[idx].clone())
            .collect();
        Arc::new(Dataset {
            events: bootstrapped_events,
        })
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

    /// Filter the [`Dataset`] by a given `predicate`, selecting events for which the predicate
    /// returns `true`.
    pub fn filter<P>(&self, predicate: P) -> Arc<Dataset>
    where
        P: Fn(&Event) -> bool + Send + Sync,
    {
        #[cfg(feature = "rayon")]
        let filtered_events = self
            .events
            .par_iter()
            .filter(|e| predicate(e))
            .cloned()
            .collect();
        #[cfg(not(feature = "rayon"))]
        let filtered_events = self
            .events
            .iter()
            .filter(|e| predicate(e))
            .cloned()
            .collect();
        Arc::new(Dataset {
            events: filtered_events,
        })
    }

    /// Bin a [`Dataset`] by the value of the given [`Variable`] into a number of `bins` within the
    /// given `range`.
    pub fn bin_by<V>(&self, variable: V, bins: usize, range: (Float, Float)) -> BinnedDataset
    where
        V: Variable,
    {
        let bin_width = (range.1 - range.0) / bins as Float;
        let bin_edges = get_bin_edges(bins, range);
        #[cfg(feature = "rayon")]
        let evaluated: Vec<(usize, &Arc<Event>)> = self
            .events
            .par_iter()
            .filter_map(|event| {
                let value = variable.value(event.as_ref());
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, event))
                } else {
                    None
                }
            })
            .collect();
        #[cfg(not(feature = "rayon"))]
        let evaluated: Vec<(usize, &Arc<Event>)> = self
            .events
            .iter()
            .filter_map(|event| {
                let value = variable.value(event.as_ref());
                if value >= range.0 && value < range.1 {
                    let bin_index = ((value - range.0) / bin_width) as usize;
                    let bin_index = bin_index.min(bins - 1);
                    Some((bin_index, event))
                } else {
                    None
                }
            })
            .collect();
        let mut binned_events: Vec<Vec<Arc<Event>>> = vec![Vec::default(); bins];
        for (bin_index, event) in evaluated {
            binned_events[bin_index].push(event.clone());
        }
        BinnedDataset {
            #[cfg(feature = "rayon")]
            datasets: binned_events
                .into_par_iter()
                .map(|events| Arc::new(Dataset { events }))
                .collect(),
            #[cfg(not(feature = "rayon"))]
            datasets: binned_events
                .into_iter()
                .map(|events| Arc::new(Dataset { events }))
                .collect(),
            edges: bin_edges,
        }
    }

    /// Boost all the four-momenta in all [`Event`]s to the rest frame of the given set of
    /// four-momenta by indices.
    pub fn boost_to_rest_frame_of<T: AsRef<[usize]> + Sync>(&self, indices: T) -> Arc<Dataset> {
        #[cfg(feature = "rayon")]
        {
            Arc::new(Dataset {
                events: self
                    .events
                    .par_iter()
                    .map(|event| Arc::new(event.boost_to_rest_frame_of(indices.as_ref())))
                    .collect(),
            })
        }
        #[cfg(not(feature = "rayon"))]
        {
            Arc::new(Dataset {
                events: self
                    .events
                    .iter()
                    .map(|event| Arc::new(event.boost_to_rest_frame_of(indices.as_ref())))
                    .collect(),
            })
        }
    }
}

impl_op_ex!(+ |a: &Dataset, b: &Dataset| ->  Dataset { Dataset { events: a.events.iter().chain(b.events.iter()).cloned().collect() }});

fn batch_to_event(batch: &RecordBatch, row: usize) -> Event {
    let mut p4s = Vec::new();
    let mut aux = Vec::new();

    let p4_count = batch
        .schema()
        .fields()
        .iter()
        .filter(|field| field.name().starts_with(P4_PREFIX))
        .count()
        / 4;
    let aux_count = batch
        .schema()
        .fields()
        .iter()
        .filter(|field| field.name().starts_with(AUX_PREFIX))
        .count()
        / 3;

    for i in 0..p4_count {
        let e = batch
            .column_by_name(&format!("{}{}_E", P4_PREFIX, i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let px = batch
            .column_by_name(&format!("{}{}_Px", P4_PREFIX, i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let py = batch
            .column_by_name(&format!("{}{}_Py", P4_PREFIX, i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let pz = batch
            .column_by_name(&format!("{}{}_Pz", P4_PREFIX, i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        p4s.push(Vec4::new(px, py, pz, e));
    }

    // TODO: insert empty vectors if not provided
    for i in 0..aux_count {
        let x = batch
            .column_by_name(&format!("{}{}_x", AUX_PREFIX, i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let y = batch
            .column_by_name(&format!("{}{}_y", AUX_PREFIX, i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let z = batch
            .column_by_name(&format!("{}{}_z", AUX_PREFIX, i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        aux.push(Vec3::new(x, y, z));
    }

    let weight = batch
        .column(19)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(row) as Float;

    Event { p4s, aux, weight }
}

/// Open a Parquet file and read the data into a [`Dataset`].
pub fn open<T: AsRef<str>>(file_path: T) -> Result<Arc<Dataset>, LadduError> {
    // TODO: make this read in directly to MPI ranks
    let file_path = Path::new(&*shellexpand::full(file_path.as_ref())?).canonicalize()?;
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

    #[cfg(feature = "rayon")]
    let events: Vec<Arc<Event>> = batches
        .into_par_iter()
        .flat_map(|batch| {
            let num_rows = batch.num_rows();
            let mut local_events = Vec::with_capacity(num_rows);

            // Process each row in the batch
            for row in 0..num_rows {
                let event = batch_to_event(&batch, row);
                local_events.push(Arc::new(event));
            }
            local_events
        })
        .collect();
    #[cfg(not(feature = "rayon"))]
    let events: Vec<Arc<Event>> = batches
        .into_iter()
        .flat_map(|batch| {
            let num_rows = batch.num_rows();
            let mut local_events = Vec::with_capacity(num_rows);

            // Process each row in the batch
            for row in 0..num_rows {
                let event = batch_to_event(&batch, row);
                local_events.push(Arc::new(event));
            }
            local_events
        })
        .collect();
    Ok(Arc::new(Dataset::new(events)))
}

/// A list of [`Dataset`]s formed by binning [`Event`]s by some [`Variable`].
pub struct BinnedDataset {
    datasets: Vec<Arc<Dataset>>,
    edges: Vec<Float>,
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
    pub fn edges(&self) -> Vec<Float> {
        self.edges.clone()
    }

    /// Returns the range that was used to form the [`BinnedDataset`].
    pub fn range(&self) -> (Float, Float) {
        (self.edges[0], self.edges[self.n_bins()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{assert_relative_eq, assert_relative_ne};
    use serde::{Deserialize, Serialize};
    #[test]
    fn test_event_creation() {
        let event = test_event();
        assert_eq!(event.p4s.len(), 4);
        assert_eq!(event.aux.len(), 1);
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
    fn test_dataset_size_check() {
        let mut dataset = Dataset::default();
        assert_eq!(dataset.n_events(), 0);
        dataset.events.push(Arc::new(test_event()));
        assert_eq!(dataset.n_events(), 1);
    }

    #[test]
    fn test_dataset_sum() {
        let dataset = test_dataset();
        let dataset2 = Dataset::new(vec![Arc::new(Event {
            p4s: test_event().p4s,
            aux: test_event().aux,
            weight: 0.52,
        })]);
        let dataset_sum = &dataset + &dataset2;
        assert_eq!(dataset_sum[0].weight, dataset[0].weight);
        assert_eq!(dataset_sum[1].weight, dataset2[0].weight);
    }

    #[test]
    fn test_dataset_weights() {
        let mut dataset = Dataset::default();
        dataset.events.push(Arc::new(test_event()));
        dataset.events.push(Arc::new(Event {
            p4s: test_event().p4s,
            aux: test_event().aux,
            weight: 0.52,
        }));
        let weights = dataset.weights();
        assert_eq!(weights.len(), 2);
        assert_relative_eq!(weights[0], 0.48);
        assert_relative_eq!(weights[1], 0.52);
        assert_relative_eq!(dataset.n_events_weighted(), 1.0);
    }

    #[test]
    fn test_dataset_filtering() {
        let mut dataset = test_dataset();
        dataset.events.push(Arc::new(Event {
            p4s: vec![
                Vec3::new(0.0, 0.0, 5.0).with_mass(0.0),
                Vec3::new(0.0, 0.0, 1.0).with_mass(1.0),
            ],
            aux: vec![],
            weight: 1.0,
        }));

        let filtered = dataset.filter(|event| event.p4s.len() == 2);
        assert_eq!(filtered.n_events(), 1);
        assert_eq!(filtered[0].p4s.len(), 2);
    }

    #[test]
    fn test_binned_dataset() {
        let dataset = Dataset::new(vec![
            Arc::new(Event {
                p4s: vec![Vec3::new(0.0, 0.0, 1.0).with_mass(1.0)],
                aux: vec![],
                weight: 1.0,
            }),
            Arc::new(Event {
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
            fn value(&self, event: &Event) -> Float {
                event.p4s[0].e()
            }
        }
        assert_eq!(BeamEnergy.to_string(), "BeamEnergy");

        // Test binning by first particle energy
        let binned = dataset.bin_by(BeamEnergy, 2, (0.0, 3.0));

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
        let mut dataset = test_dataset();
        dataset.events.push(Arc::new(Event {
            p4s: test_event().p4s.clone(),
            aux: test_event().aux.clone(),
            weight: 1.0,
        }));
        assert_relative_ne!(dataset[0].weight, dataset[1].weight);

        let bootstrapped = dataset.bootstrap(43);
        assert_eq!(bootstrapped.n_events(), dataset.n_events());
        assert_relative_eq!(bootstrapped[0].weight, bootstrapped[1].weight);

        // Test empty dataset bootstrap
        let empty_dataset = Dataset::default();
        let empty_bootstrap = empty_dataset.bootstrap(43);
        assert_eq!(empty_bootstrap.n_events(), 0);
    }

    #[test]
    fn test_event_display() {
        let event = test_event();
        let display_string = format!("{}", event);
        assert_eq!(
            display_string,
            "Event:\n  p4s:\n    [e = 8.74700; p = (0.00000, 0.00000, 8.74700); m = 0.00000]\n    [e = 1.10334; p = (0.11900, 0.37400, 0.22200); m = 1.00700]\n    [e = 3.13671; p = (-0.11200, 0.29300, 3.08100); m = 0.49800]\n    [e = 5.50925; p = (-0.00700, -0.66700, 5.44600); m = 0.49800]\n  eps:\n    [0.385, 0.022, 0]\n  weight:\n    0.48\n"
        );
    }
}
