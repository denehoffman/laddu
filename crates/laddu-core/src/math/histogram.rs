/// A helper method to get histogram edges from evenly-spaced `bins` over a given `range`
///
/// # See Also
/// [`Histogram`]
/// [`get_bin_index`]
///
/// # Examples
/// ```rust
/// use laddu_core::math::get_bin_edges;
///
/// assert_eq!(get_bin_edges(3, (0.0, 3.0)), vec![0.0, 1.0, 2.0, 3.0]);
/// ```
pub fn get_bin_edges(bins: usize, range: (f64, f64)) -> Vec<f64> {
    let bin_width = (range.1 - range.0) / (bins as f64);
    (0..=bins)
        .map(|i| range.0 + (i as f64 * bin_width))
        .collect()
}

/// A helper method to obtain the index of a bin where a value should go in a histogram with evenly
/// spaced `bins` over a given `range`
///
/// # See Also
/// [`Histogram`]
/// [`get_bin_edges`]
///
/// # Examples
/// ```rust
/// use laddu_core::math::get_bin_index;
///
/// assert_eq!(get_bin_index(0.25, 4, (0.0, 1.0)), Some(1));
/// assert_eq!(get_bin_index(1.5, 4, (0.0, 1.0)), None);
/// ```
pub fn get_bin_index(value: f64, bins: usize, range: (f64, f64)) -> Option<usize> {
    if value >= range.0 && value < range.1 {
        let bin_width = (range.1 - range.0) / bins as f64;
        let bin_index = ((value - range.0) / bin_width).floor() as usize;
        Some(bin_index.min(bins - 1))
    } else {
        None
    }
}

use serde::{Deserialize, Serialize};

use crate::{LadduError, LadduResult};

/// A simple weighted histogram with explicit bin edges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    /// The number of counts in each bin (can be [`f64`]s since these might be weighted counts)
    pub counts: Vec<f64>,
    /// The edges of each bin (length is one greater than `counts`)
    pub bin_edges: Vec<f64>,
}

impl Histogram {
    /// Construct and validate a histogram from bin edges and weighted bin counts.
    pub fn new(bin_edges: Vec<f64>, counts: Vec<f64>) -> LadduResult<Self> {
        let histogram = Self { counts, bin_edges };
        histogram.validate()?;
        Ok(histogram)
    }

    /// Return the number of weighted counts in each bin.
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// Return the bin edges.
    pub fn bin_edges(&self) -> &[f64] {
        &self.bin_edges
    }

    /// Return the total histogram weight.
    pub fn total_weight(&self) -> f64 {
        self.counts.iter().sum()
    }

    /// Validate histogram asserting each bin has nonnegative counts.
    pub fn validate_positive_counts(&self) -> LadduResult<()> {
        for (index, count) in self.counts.iter().enumerate() {
            if *count < 0.0 {
                return Err(LadduError::Custom(format!(
                    "histogram count {index} must be nonnegative, got {count}"
                )));
            }
        }
        Ok(())
    }

    /// Validate histogram shape, finite values, and positive integral.
    pub fn validate(&self) -> LadduResult<()> {
        if self.bin_edges.len() < 2 {
            return Err(LadduError::Custom(
                "histogram requires at least two bin edges".to_string(),
            ));
        }
        if self.counts.len() + 1 != self.bin_edges.len() {
            return Err(LadduError::Custom(format!(
                "histogram requires counts.len() + 1 == bin_edges.len(), got {} counts and {} edges",
                self.counts.len(),
                self.bin_edges.len()
            )));
        }
        for (index, edge) in self.bin_edges.iter().enumerate() {
            if !edge.is_finite() {
                return Err(LadduError::Custom(format!(
                    "histogram bin edge {index} must be finite, got {edge}"
                )));
            }
        }
        for (index, edges) in self.bin_edges.windows(2).enumerate() {
            if edges[1] <= edges[0] {
                return Err(LadduError::Custom(format!(
                    "histogram bin edges must be strictly increasing at edge pair {index}"
                )));
            }
        }
        for (index, count) in self.counts.iter().enumerate() {
            if !count.is_finite() {
                return Err(LadduError::Custom(format!(
                    "histogram count {index} must be finite, got {count}"
                )));
            }
        }
        let total_weight = self.total_weight();
        if total_weight <= 0.0 {
            return Err(LadduError::Custom(format!(
                "histogram total weight must be positive, got {total_weight}"
            )));
        }
        Ok(())
    }
}

/// A method which creates a histogram from some data by binning it with evenly spaced `bins`
/// within the given `range`
///
/// # Examples
/// ```rust
/// use laddu_core::math::histogram;
///
/// let values = vec![0.1, 0.4, 0.8];
/// let weights: Option<&[f64]> = None;
/// let hist = histogram(values.as_slice(), 2, (0.0, 1.0), weights);
/// assert_eq!(hist.counts, vec![2.0, 1.0]);
/// assert_eq!(hist.bin_edges, vec![0.0, 0.5, 1.0]);
/// ```
pub fn histogram<T: AsRef<[f64]>>(
    values: T,
    bins: usize,
    range: (f64, f64),
    weights: Option<T>,
) -> Histogram {
    assert!(bins > 0, "Number of bins must be greater than zero!");
    assert!(
        range.1 > range.0,
        "The lower edge of the range must be smaller than the upper edge!"
    );
    if let Some(w) = &weights {
        assert_eq!(
            values.as_ref().len(),
            w.as_ref().len(),
            "`values` and `weights` must have the same length!"
        );
    }
    let mut counts = vec![0.0; bins];
    for (i, &value) in values.as_ref().iter().enumerate() {
        if let Some(bin_index) = get_bin_index(value, bins, range) {
            let weight = weights.as_ref().map_or(1.0, |w| w.as_ref()[i]);
            counts[bin_index] += weight;
        }
    }
    Histogram::new(get_bin_edges(bins, range), counts)
        .expect("histogram helper should construct a valid histogram")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{get_bin_index, histogram, Histogram};
    use crate::{data::test_dataset, traits::Variable, Mass};

    #[test]
    fn test_binning() {
        let mut v = Mass::new(["kshort1"]);
        let dataset = Arc::new(test_dataset());
        v.bind(dataset.metadata()).unwrap();
        let values = v.value_on(&dataset).unwrap();
        let bin_index = get_bin_index(values[0], 3, (0.0, 1.0));
        assert_eq!(bin_index, Some(1));
        let bin_index = get_bin_index(0.0, 3, (0.0, 1.0));
        assert_eq!(bin_index, Some(0));
        let bin_index = get_bin_index(0.1, 3, (0.0, 1.0));
        assert_eq!(bin_index, Some(0));
        let bin_index = get_bin_index(0.9, 3, (0.0, 1.0));
        assert_eq!(bin_index, Some(2));
        let bin_index = get_bin_index(1.0, 3, (0.0, 1.0));
        assert_eq!(bin_index, None);
        let bin_index = get_bin_index(2.0, 3, (0.0, 1.0));
        assert_eq!(bin_index, None);
        let weights = dataset.weights();
        let histogram = histogram(&values, 3, (0.0, 1.0), Some(&weights));
        assert_eq!(histogram.counts, vec![0.0, 0.48, 0.0]);
        assert_eq!(histogram.bin_edges, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    }

    #[test]
    fn histogram_new_validates_shape_and_values() {
        assert!(Histogram::new(vec![0.0, 1.0], vec![1.0]).is_ok());
        assert!(Histogram::new(vec![0.0], vec![]).is_err());
        assert!(Histogram::new(vec![0.0, 1.0], vec![1.0, 2.0]).is_err());
        assert!(Histogram::new(vec![0.0, 0.0], vec![1.0]).is_err());
        assert!(Histogram::new(vec![0.0, f64::NAN], vec![1.0]).is_err());
        assert!(Histogram::new(vec![0.0, 1.0], vec![-1.0]).is_err());
        assert!(Histogram::new(vec![0.0, 1.0], vec![0.0]).is_err());
    }

    #[test]
    fn histogram_serializes() {
        let histogram = Histogram::new(vec![0.0, 1.0, 2.0], vec![1.0, 2.0]).unwrap();
        let serialized = serde_pickle::to_vec(&histogram, serde_pickle::SerOptions::new()).unwrap();
        let restored: Histogram =
            serde_pickle::from_slice(&serialized, serde_pickle::DeOptions::new()).unwrap();
        assert_eq!(restored, histogram);
    }
}
