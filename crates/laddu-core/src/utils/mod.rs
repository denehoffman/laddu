/// Useful enumerations for various frames and variables common in particle physics analyses.
pub mod enums;
/// Standard special functions like spherical harmonics and momentum definitions.
pub mod functions;
/// Traits and structs which can be used to extract complex information from
/// [`EventData`](crate::data::EventData)s.
pub mod variables;
/// Traits to give additional functionality to [`nalgebra::Vector3`] and [`nalgebra::Vector4`] (in
/// particular, to treat the latter as a four-momentum).
pub mod vectors;

/// A helper method to get histogram edges from evenly-spaced `bins` over a given `range`
///
/// # See Also
/// [`Histogram`]
/// [`get_bin_index`]
///
/// # Examples
/// ```rust
/// use laddu_core::utils::get_bin_edges;
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
/// use laddu_core::utils::get_bin_index;
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

/// A simple struct which represents a histogram
pub struct Histogram {
    /// The number of counts in each bin (can be [`f64`]s since these might be weighted counts)
    pub counts: Vec<f64>,
    /// The edges of each bin (length is one greater than `counts`)
    pub bin_edges: Vec<f64>,
}

/// A method which creates a histogram from some data by binning it with evenly spaced `bins` within
/// the given `range`
///
/// # Examples
/// ```rust
/// use laddu_core::utils::histogram;
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
    Histogram {
        counts,
        bin_edges: get_bin_edges(bins, range),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        data::test_dataset,
        traits::Variable,
        utils::{get_bin_index, histogram},
        Mass,
    };

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
}
