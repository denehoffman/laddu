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

pub(crate) fn get_bin_index(value: Float, bin_edges: &[Float]) -> Option<usize> {
    if value < bin_edges[0] || value >= bin_edges[bin_edges.len() - 1] {
        return None;
    }
    bin_edges
        .windows(2)
        .position(|edges| value >= edges[0] && value < edges[1])
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        data::test_dataset,
        traits::Variable,
        utils::{get_bin_edges, get_bin_index},
        Mass,
    };

    #[test]
    fn test_binning() {
        let v = Mass::new([2]);
        let dataset = Arc::new(test_dataset());
        let bin_edges = get_bin_edges(3, (0.0, 1.0));
        let bin_index = get_bin_index(v.value_on(&dataset)[0], &bin_edges);
        assert_eq!(bin_index, Some(1));
        let bin_index = get_bin_index(0.0, &bin_edges);
        assert_eq!(bin_index, Some(0));
        let bin_index = get_bin_index(0.1, &bin_edges);
        assert_eq!(bin_index, Some(0));
        let bin_index = get_bin_index(0.9, &bin_edges);
        assert_eq!(bin_index, Some(2));
        let bin_index = get_bin_index(1.0, &bin_edges);
        assert_eq!(bin_index, None);
        let bin_index = get_bin_index(2.0, &bin_edges);
        assert_eq!(bin_index, None);
    }
}
