use arrow::array::Float32Array;
use arrow::record_batch::RecordBatch;
use nalgebra::{vector, Vector3, Vector4};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::path::Path;
use std::sync::Arc;
use std::{fmt::Display, fs::File};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::utils::get_bin_edges;
use crate::{
    utils::{variables::Variable, vectors::FourMomentum},
    Float, LadduError,
};

/// An event that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitudes::Amplitude). This particular event contains the reaction
/// $`\gamma p \to K_S^0 K_S^0 p`$ with a polarized photon beam.
pub fn test_event() -> Event {
    use crate::utils::vectors::*;
    Event {
        p4s: vec![
            vector![0.0, 0.0, 8.747].with_mass(0.0),         // beam
            vector![0.119, 0.374, 0.222].with_mass(1.007),   // "proton"
            vector![-0.112, 0.293, 3.081].with_mass(0.498),  // "kaon"
            vector![-0.007, -0.667, 5.446].with_mass(0.498), // "kaon"
        ],
        eps: vec![vector![0.385, 0.022, 0.000]],
        weight: 0.48,
    }
}

/// An dataset that can be used to test the implementation of an
/// [`Amplitude`](crate::amplitudes::Amplitude). This particular dataset contains a singular
/// [`Event`] generated from [`test_event`].
pub fn test_dataset() -> Dataset {
    Dataset {
        events: vec![Arc::new(test_event())],
    }
}

/// A single event in a [`Dataset`] containing all the relevant particle information.
#[derive(Debug, Clone, Default)]
pub struct Event {
    /// A list of four-momenta for each particle.
    pub p4s: Vec<Vector4<Float>>,
    /// A list of polarization vectors for each particle.
    pub eps: Vec<Vector3<Float>>,
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
        for eps_vec in &self.eps {
            writeln!(
                f,
                "    [{}]",
                eps_vec
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            )?;
        }
        writeln!(f, "  weight:")?;
        writeln!(f, "    {}", self.weight)?;
        Ok(())
    }
}

impl Event {
    /// Return a four-momentum from the sum of four-momenta at the given indices in the [`Event`].
    pub fn get_p4_sum<T: AsRef<[usize]>>(&self, indices: T) -> Vector4<Float> {
        indices
            .as_ref()
            .iter()
            .map(|i| self.p4s[*i])
            .sum::<Vector4<Float>>()
    }
}

/// A collection of [`Event`]s.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// The [`Event`]s contained in the [`Dataset`]
    pub events: Vec<Arc<Event>>,
}

impl Index<usize> for Dataset {
    type Output = Event;

    fn index(&self, index: usize) -> &Self::Output {
        &self.events[index]
    }
}

impl Deref for Dataset {
    type Target = Vec<Arc<Event>>;

    fn deref(&self) -> &Self::Target {
        &self.events
    }
}

impl DerefMut for Dataset {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.events
    }
}

impl Dataset {
    /// The number of [`Event`]s in the [`Dataset`].
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Checks whether or not the [`Dataset`] is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Produces an iterator over the [`Event`]s in the [`Dataset`].
    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter().map(|a| a.as_ref())
    }
}

#[cfg(feature = "rayon")]
impl Dataset {
    /// Produces an parallelized iterator over the [`Event`]s in the [`Dataset`].
    pub fn par_iter(&self) -> impl IndexedParallelIterator<Item = &Event> {
        self.events.par_iter().map(|a| a.as_ref())
    }

    /// Extract a list of weights over each [`Event`] in the [`Dataset`].
    pub fn weights(&self) -> Vec<Float> {
        self.par_iter().map(|e| e.weight).collect()
    }

    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`].
    pub fn weighted_len(&self) -> Float {
        self.par_iter().map(|e| e.weight).sum()
    }

    /// Generate a new dataset with the same length by resampling the events in the original datset
    /// with replacement. This can be used to perform error analysis via the bootstrap method.
    pub fn bootstrap(&self, seed: usize) -> Arc<Dataset> {
        if self.is_empty() {
            return Arc::new(Dataset::default());
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let mut indices: Vec<usize> = (0..self.len())
            .map(|_| rng.gen_range(0..self.len()))
            .collect::<Vec<usize>>();
        indices.sort();
        let bootstrapped_events: Vec<Arc<Event>> = indices
            .into_par_iter()
            .map(|idx| self.events[idx].clone())
            .collect();
        Arc::new(Dataset {
            events: bootstrapped_events,
        })
    }

    /// Filter the [`Dataset`] by a given `predicate`, selecting events for which the predicate
    /// returns `true`.
    pub fn filter<P>(&self, predicate: P) -> Arc<Dataset>
    where
        P: Fn(&Event) -> bool + Send + Sync,
    {
        let filtered_events = self
            .events
            .par_iter()
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
        let mut binned_events: Vec<Vec<Arc<Event>>> = vec![Vec::default(); bins];
        for (bin_index, event) in evaluated {
            binned_events[bin_index].push(event.clone());
        }
        BinnedDataset {
            datasets: binned_events
                .into_par_iter()
                .map(|events| Arc::new(Dataset { events }))
                .collect(),
            edges: bin_edges,
        }
    }
}

#[cfg(not(feature = "rayon"))]
impl Dataset {
    /// Extract a list of weights over each [`Event`] in the [`Dataset`].
    pub fn weights(&self) -> Vec<Float> {
        self.iter().map(|e| e.weight).collect()
    }

    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`].
    pub fn weighted_len(&self) -> Float {
        self.iter().map(|e| e.weight).sum()
    }

    /// Generate a new dataset with the same length by resampling the events in the original datset
    /// with replacement. This can be used to perform error analysis via the bootstrap method.
    pub fn bootstrap(&self, seed: usize) -> Arc<Dataset> {
        if self.is_empty() {
            return Arc::new(Dataset::default());
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let mut indices: Vec<usize> = (0..self.len())
            .map(|_| rng.gen_range(0..self.len()))
            .collect::<Vec<usize>>();
        indices.sort();
        let bootstrapped_events: Vec<Arc<Event>> = indices
            .into_iter()
            .map(|idx| self.events[idx].clone())
            .collect();
        Arc::new(Dataset {
            events: bootstrapped_events,
        })
    }

    /// Filter the [`Dataset`] by a given `predicate`, selecting events for which the predicate
    /// returns `true`.
    pub fn filter<P>(&self, predicate: P) -> Arc<Dataset>
    where
        P: Fn(&Event) -> bool + Send + Sync,
    {
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
            datasets: binned_events
                .into_iter()
                .map(|events| Arc::new(Dataset { events }))
                .collect(),
            edges: bin_edges,
        }
    }
}

fn batch_to_event(batch: &RecordBatch, row: usize) -> Event {
    let mut p4s = Vec::new();
    let mut eps = Vec::new();

    let p4_count = batch
        .schema()
        .fields()
        .iter()
        .filter(|field| field.name().starts_with("p4_"))
        .count()
        / 4;
    let eps_count = batch
        .schema()
        .fields()
        .iter()
        .filter(|field| field.name().starts_with("eps_"))
        .count()
        / 3;

    for i in 0..p4_count {
        let e = batch
            .column_by_name(&format!("p4_{}_E", i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let px = batch
            .column_by_name(&format!("p4_{}_Px", i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let py = batch
            .column_by_name(&format!("p4_{}_Py", i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let pz = batch
            .column_by_name(&format!("p4_{}_Pz", i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        p4s.push(Vector4::new(px, py, pz, e));
    }

    // TODO: insert empty vectors if not provided
    for i in 0..eps_count {
        let x = batch
            .column_by_name(&format!("eps_{}_x", i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let y = batch
            .column_by_name(&format!("eps_{}_y", i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        let z = batch
            .column_by_name(&format!("eps_{}_z", i))
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .value(row) as Float;
        eps.push(Vector3::new(x, y, z));
    }

    let weight = batch
        .column(19)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap()
        .value(row) as Float;

    Event { p4s, eps, weight }
}

/// Open a Parquet file and read the data into a [`Dataset`].
#[cfg(feature = "rayon")]
pub fn open<T: AsRef<str>>(file_path: T) -> Result<Arc<Dataset>, LadduError> {
    let file_path = Path::new(&*shellexpand::full(file_path.as_ref())?).canonicalize()?;
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

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
    Ok(Arc::new(Dataset { events }))
}

/// Open a Parquet file and read the data into a [`Dataset`].
#[cfg(not(feature = "rayon"))]
pub fn open(file_path: &str) -> Result<Arc<Dataset>, LadduError> {
    let file_path = Path::new(&*shellexpand::full(file_path)?).canonicalize()?;
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

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
    Ok(Arc::new(Dataset { events }))
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
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Checks whether or not the [`BinnedDataset`] is empty.
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// The number of bins in the [`BinnedDataset`]. Alias of [`BinnedDataset::len()`].
    pub fn bins(&self) -> usize {
        self.len()
    }

    /// Returns a list of the bin edges that were used to form the [`BinnedDataset`].
    pub fn edges(&self) -> Vec<Float> {
        self.edges.clone()
    }

    /// Returns the range that was used to form the [`BinnedDataset`].
    pub fn range(&self) -> (Float, Float) {
        (self.edges[0], self.edges[self.len()])
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::ThreeMomentum;

    use super::*;
    use approx::{assert_relative_eq, assert_relative_ne};
    use serde::{Deserialize, Serialize};
    #[test]
    fn test_event_creation() {
        let event = test_event();
        assert_eq!(event.p4s.len(), 4);
        assert_eq!(event.eps.len(), 1);
        assert_relative_eq!(event.weight, 0.48)
    }

    #[test]
    fn test_event_p4_sum() {
        let event = test_event();
        let sum = event.get_p4_sum([2, 3]);
        assert_relative_eq!(sum[0], event.p4s[2].px() + event.p4s[3].px());
        assert_relative_eq!(sum[1], event.p4s[2].py() + event.p4s[3].py());
        assert_relative_eq!(sum[2], event.p4s[2].pz() + event.p4s[3].pz());
        assert_relative_eq!(sum[3], event.p4s[2].e() + event.p4s[3].e());
    }

    #[test]
    fn test_dataset_size_check() {
        let mut dataset = Dataset::default();
        assert!(dataset.is_empty());
        assert_eq!(dataset.len(), 0);
        dataset.events.push(Arc::new(test_event()));
        assert!(!dataset.is_empty());
        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_dataset_weights() {
        let mut dataset = Dataset::default();
        dataset.events.push(Arc::new(test_event()));
        dataset.events.push(Arc::new(Event {
            p4s: test_event().p4s,
            eps: test_event().eps,
            weight: 0.52,
        }));
        let weights = dataset.weights();
        assert_eq!(weights.len(), 2);
        assert_relative_eq!(weights[0], 0.48);
        assert_relative_eq!(weights[1], 0.52);
        assert_relative_eq!(dataset.weighted_len(), 1.0);
    }

    #[test]
    fn test_dataset_filtering() {
        let mut dataset = test_dataset();
        dataset.events.push(Arc::new(Event {
            p4s: vec![
                vector![0.0, 0.0, 5.0].with_mass(0.0),
                vector![0.0, 0.0, 1.0].with_mass(1.0),
            ],
            eps: vec![],
            weight: 1.0,
        }));

        let filtered = dataset.filter(|event| event.p4s.len() == 2);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].p4s.len(), 2);
    }

    #[test]
    fn test_binned_dataset() {
        let mut dataset = Dataset::default();
        dataset.events.push(Arc::new(Event {
            p4s: vec![vector![0.0, 0.0, 1.0].with_mass(1.0)],
            eps: vec![],
            weight: 1.0,
        }));
        dataset.events.push(Arc::new(Event {
            p4s: vec![vector![0.0, 0.0, 2.0].with_mass(2.0)],
            eps: vec![],
            weight: 2.0,
        }));

        #[derive(Clone, Serialize, Deserialize)]
        struct BeamEnergy;
        #[typetag::serde]
        impl Variable for BeamEnergy {
            fn value(&self, event: &Event) -> Float {
                event.p4s[0].e()
            }
        }

        // Test binning by first particle energy
        let binned = dataset.bin_by(BeamEnergy, 2, (0.0, 3.0));

        assert_eq!(binned.bins(), 2);
        assert_eq!(binned.edges().len(), 3);
        assert_relative_eq!(binned.edges()[0], 0.0);
        assert_relative_eq!(binned.edges()[2], 3.0);
        assert_eq!(binned[0].len(), 1);
        assert_relative_eq!(binned[0].weighted_len(), 1.0);
        assert_eq!(binned[1].len(), 1);
        assert_relative_eq!(binned[1].weighted_len(), 2.0);
    }

    #[test]
    fn test_dataset_bootstrap() {
        let mut dataset = test_dataset();
        dataset.events.push(Arc::new(Event {
            p4s: test_event().p4s.clone(),
            eps: test_event().eps.clone(),
            weight: 1.0,
        }));
        assert_relative_ne!(dataset[0].weight, dataset[1].weight);

        let bootstrapped = dataset.bootstrap(42);
        assert_eq!(bootstrapped.len(), dataset.len());
        assert_relative_eq!(bootstrapped[0].weight, bootstrapped[1].weight);

        // Test empty dataset bootstrap
        let empty_dataset = Dataset::default();
        let empty_bootstrap = empty_dataset.bootstrap(42);
        assert!(empty_bootstrap.is_empty());
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
