use arrow::array::Float32Array;
use arrow::record_batch::RecordBatch;
use nalgebra::{vector, Vector3, Vector4};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::path::Path;
use std::sync::Arc;
use std::{fmt::Display, fs::File};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

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
            Vector4::from_momentum(&vector![0.0, 0.0, 8.747], 0.0), // beam
            Vector4::from_momentum(&vector![0.119, 0.374, 0.222], 1.007), // "proton"
            Vector4::from_momentum(&vector![-0.112, 0.293, 3.081], 0.498), // "kaon"
            Vector4::from_momentum(&vector![-0.007, -0.667, 5.446], 0.498), // "kaon"
        ],
        eps: vec![vector![0.385, 0.022, 0.000]],
        weight: 0.48,
    }
}

/// A single event in a [`Dataset`] containing all the relevant particle information.
#[derive(Debug, Clone)]
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
    pub fn get_p4_sum(&self, indices: &[usize]) -> Vector4<Float> {
        indices.iter().map(|i| self.p4s[*i]).sum::<Vector4<Float>>()
    }
}

/// A collection of [`Event`]s.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub(crate) events: Vec<Event>,
}

impl Index<usize> for Dataset {
    type Output = Event;

    fn index(&self, index: usize) -> &Self::Output {
        &self.events[index]
    }
}

impl IndexMut<usize> for Dataset {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.events[index]
    }
}

impl Deref for Dataset {
    type Target = Vec<Event>;

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
    pub fn iter(&self) -> std::slice::Iter<'_, Event> {
        self.events.iter()
    }

    /// Produces an parallelized iterator over the [`Event`]s in the [`Dataset`].
    #[cfg(feature = "rayon")]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, Event> {
        self.events.par_iter()
    }

    /// Extract a list of weights over each [`Event`] in the [`Dataset`].
    #[cfg(not(feature = "rayon"))]
    pub fn weights(&self) -> Vec<Float> {
        self.iter().map(|e| e.weight).collect()
    }

    /// Extract a list of weights over each [`Event`] in the [`Dataset`].
    #[cfg(feature = "rayon")]
    pub fn weights(&self) -> Vec<Float> {
        self.par_iter().map(|e| e.weight).collect()
    }

    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`].
    #[cfg(not(feature = "rayon"))]
    pub fn weighted_len(&self) -> Float {
        self.iter().map(|e| e.weight).sum()
    }

    /// Returns the sum of the weights for each [`Event`] in the [`Dataset`].
    #[cfg(feature = "rayon")]
    pub fn weighted_len(&self) -> Float {
        self.par_iter().map(|e| e.weight).sum()
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
        p4s.push(Vector4::new(e, px, py, pz));
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
pub fn open(file_path: &str) -> Result<Arc<Dataset>, LadduError> {
    let file_path = Path::new(&*shellexpand::full(file_path)?).canonicalize()?;
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

    let events: Vec<Event> = batches
        .into_par_iter()
        .flat_map(|batch| {
            let num_rows = batch.num_rows();
            let mut local_events = Vec::with_capacity(num_rows);

            // Process each row in the batch
            for row in 0..num_rows {
                let event = batch_to_event(&batch, row);
                local_events.push(event);
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

    let events: Vec<Event> = batches
        .into_iter()
        .flat_map(|batch| {
            let num_rows = batch.num_rows();
            let mut local_events = Vec::with_capacity(num_rows);

            // Process each row in the batch
            for row in 0..num_rows {
                let event = batch_to_event(&batch, row);
                local_events.push(event);
            }
            local_events
        })
        .collect();
    Ok(Arc::new(Dataset { events }))
}

/// Open a Parquet file and read the data into a [`Dataset`]. Only returns events for which the
/// given predicate returns `true`.
#[cfg(feature = "rayon")]
pub fn open_filtered<P>(file_path: &str, predicate: P) -> Result<Arc<Dataset>, LadduError>
where
    P: Fn(&Event) -> bool + Send + Sync,
{
    let file_path = Path::new(&*shellexpand::full(file_path)?).canonicalize()?;
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

    let events: Vec<Event> = batches
        .into_par_iter()
        .flat_map(|batch| {
            let num_rows = batch.num_rows();
            let mut local_events = Vec::with_capacity(num_rows);

            // Process each row in the batch
            for row in 0..num_rows {
                let event = batch_to_event(&batch, row);
                if predicate(&event) {
                    local_events.push(event);
                }
            }
            local_events
        })
        .collect();
    Ok(Arc::new(Dataset { events }))
}

/// Open a Parquet file and read the data into a [`Dataset`]. Only returns events for which the
/// given predicate returns `true`.
#[cfg(not(feature = "rayon"))]
pub fn open_filtered<P>(file_path: &str, predicate: P) -> Result<Arc<Dataset>, LadduError>
where
    P: Fn(&Event) -> bool,
{
    let file_path = Path::new(&*shellexpand::full(file_path)?).canonicalize()?;
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

    let events: Vec<Event> = batches
        .into_iter()
        .flat_map(|batch| {
            let num_rows = batch.num_rows();
            let mut local_events = Vec::with_capacity(num_rows);

            // Process each row in the batch
            for row in 0..num_rows {
                let event = batch_to_event(&batch, row);
                if predicate(&event) {
                    local_events.push(event);
                }
            }
            local_events
        })
        .collect();
    Ok(Arc::new(Dataset { events }))
}

fn get_bin_edges(bins: usize, range: (Float, Float)) -> Vec<Float> {
    let bin_width = (range.1 - range.0) / (bins as Float);
    (0..=bins)
        .map(|i| range.0 + (i as Float * bin_width))
        .collect()
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

/// Open a Parquet file and read the data into a [`BinnedDataset`] using the given [`Variable`],
/// number of `bins` and `range`.
pub fn open_binned<V: Variable>(
    file_path: &str,
    variable: V,
    bins: usize,
    range: (Float, Float),
) -> Result<BinnedDataset, LadduError> {
    let file_path = Path::new(&*shellexpand::full(file_path)?).canonicalize()?;
    let file = File::open(file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>()?;

    let mut binned_events: Vec<Vec<Event>> = vec![Vec::default(); bins];
    let bin_width = (range.1 - range.0) / bins as Float;
    let bin_edges = get_bin_edges(bins, range);

    batches.into_iter().for_each(|batch| {
        let num_rows = batch.num_rows();

        // Process each row in the batch
        for row in 0..num_rows {
            let event = batch_to_event(&batch, row);
            let value = variable.value(&event);
            if value >= range.0 && value < range.1 {
                let bin_index = ((value - range.0) / bin_width) as usize;
                binned_events[bin_index].push(event);
            }
        }
    });
    Ok(BinnedDataset {
        datasets: binned_events
            .into_iter()
            .map(|events| Arc::new(Dataset { events }))
            .collect(),
        edges: bin_edges,
    })
}
