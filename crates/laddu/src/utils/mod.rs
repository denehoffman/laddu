use crate::Float;

/// Useful enumerations for various frames and variables common in particle physics analyses.
pub mod enums;
/// Standard special functions like spherical harmonics and momentum definitions.
pub mod functions;
/// Traits and structs which can be used to extract complex information from
/// [`Event`](crate::data::Event)s.
pub mod variables;
/// Traits to give additional functionality to [`nalgebra::Vector3`] and [`nalgebra::Vector4`] (in
/// particular, to treat the latter as a four-momentum).
pub mod vectors;

pub(crate) fn get_bin_edges(bins: usize, range: (Float, Float)) -> Vec<Float> {
    let bin_width = (range.1 - range.0) / (bins as Float);
    (0..=bins)
        .map(|i| range.0 + (i as Float * bin_width))
        .collect()
}

pub(crate) fn get_bin_index(value: Float, bins: usize, range: (Float, Float)) -> Option<usize> {
    if value >= range.0 && value < range.1 {
        let bin_width = (range.1 - range.0) / bins as Float;
        let bin_index = ((value - range.0) / bin_width).floor() as usize;
        Some(bin_index.min(bins - 1))
    } else {
        None
    }
}

pub(crate) struct Histogram {
    counts: Vec<Float>,
    bin_edges: Vec<Float>,
}

pub(crate) fn histogram<T: AsRef<[Float]>>(
    values: T,
    bins: usize,
    range: (Float, Float),
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
        let v = Mass::new([2]);
        let dataset = Arc::new(test_dataset());
        let bin_index = get_bin_index(v.value_on(&dataset)[0], 3, (0.0, 1.0));
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
        let histogram = histogram(v.value_on(&dataset), 3, (0.0, 1.0), Some(dataset.weights()));
        assert_eq!(histogram.counts, vec![0.0, 0.48, 0.0]);
        assert_eq!(histogram.bin_edges, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    }
}
