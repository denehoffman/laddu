//! Validated axes and shared bin-assignment semantics.

use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{LadduPhysicsError, LadduPhysicsResult};

/// Policy for a value exactly equal to an axis's final upper edge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FinalUpperEdge {
    /// Treat the final upper edge as outside the bounded axis.
    #[default]
    Exclusive,
    /// Assign the final upper edge to the last bin.
    Inclusive,
}

/// A finite, strictly increasing one-dimensional bin axis.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct BinningAxis {
    edges: Arc<[f64]>,
}

impl<'de> Deserialize<'de> for BinningAxis {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let edges = Vec::<f64>::deserialize(deserializer)?;
        Self::new(edges).map_err(serde::de::Error::custom)
    }
}

impl BinningAxis {
    /// Validate and construct an axis from explicit edges.
    ///
    /// # Errors
    /// Returns an error unless at least two finite, strictly increasing edges are supplied.
    pub fn new(edges: impl IntoIterator<Item = f64>) -> LadduPhysicsResult<Self> {
        let edges: Vec<_> = edges.into_iter().collect();
        if edges.len() < 2 {
            return Err(LadduPhysicsError::invalid_length(
                "bin axis edges",
                "at least 2",
                edges.len(),
            ));
        }
        if edges.iter().any(|edge| !edge.is_finite())
            || edges.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(LadduPhysicsError::invalid_relation(
                "bin axis edges must be finite and strictly increasing",
            ));
        }
        Ok(Self {
            edges: edges.into(),
        })
    }

    /// Construct uniformly spaced bins over finite increasing bounds.
    ///
    /// # Errors
    /// Returns an error for zero bins or non-finite/non-increasing bounds.
    pub fn uniform(count: usize, min: f64, max: f64) -> LadduPhysicsResult<Self> {
        if count == 0 || !min.is_finite() || !max.is_finite() || min >= max {
            return Err(LadduPhysicsError::invalid_relation(
                "uniform bins require a positive count and finite min < max",
            ));
        }
        let width = (max - min) / count as f64;
        Self::new((0..=count).map(|index| min + index as f64 * width))
    }

    /// Validated edge values.
    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    /// Number of bins along the axis.
    pub fn bin_count(&self) -> usize {
        self.edges.len() - 1
    }

    /// Assign a value using half-open internal bins and an explicit final-edge policy.
    pub fn index(&self, value: f64, final_upper: FinalUpperEdge) -> Option<usize> {
        if !value.is_finite() || value < self.edges[0] || value > *self.edges.last()? {
            return None;
        }
        if value == *self.edges.last()? {
            return (final_upper == FinalUpperEdge::Inclusive).then(|| self.bin_count() - 1);
        }
        let upper = self.edges.partition_point(|edge| *edge <= value);
        upper
            .checked_sub(1)
            .filter(|index| *index < self.bin_count())
    }
}

/// Return the row-major shape for ordered axes.
pub fn bin_shape(axes: &[BinningAxis]) -> Vec<usize> {
    axes.iter().map(BinningAxis::bin_count).collect()
}

/// Return the flattened bin count, or `None` when the ordered shape overflows `usize`.
pub fn checked_bin_count(axes: &[BinningAxis]) -> Option<usize> {
    axes.iter()
        .try_fold(1usize, |count, axis| count.checked_mul(axis.bin_count()))
}

/// Assign ordered coordinates to a row-major flattened bin.
///
/// Returns `None` for a rank mismatch, non-finite coordinate, or out-of-range coordinate.
pub fn flat_bin_index(
    axes: &[BinningAxis],
    coordinates: &[f64],
    final_upper: FinalUpperEdge,
) -> Option<usize> {
    if axes.len() != coordinates.len() {
        return None;
    }
    axes.iter()
        .zip(coordinates)
        .try_fold(0usize, |flat, (axis, value)| {
            flat.checked_mul(axis.bin_count())?
                .checked_add(axis.index(*value, final_upper)?)
        })
}

/// Assign one event from axis-major coordinate columns to a flattened bin.
///
/// Returns `None` for mismatched axes/columns, a missing event row, a non-finite
/// coordinate, an out-of-range coordinate, or flattened-index overflow.
pub fn flat_bin_index_for_event(
    axes: &[BinningAxis],
    coordinate_columns: &[Vec<f64>],
    event: usize,
    final_upper: FinalUpperEdge,
) -> Option<usize> {
    if axes.len() != coordinate_columns.len() {
        return None;
    }
    axes.iter()
        .zip(coordinate_columns)
        .try_fold(0usize, |flat, (axis, coordinates)| {
            flat.checked_mul(axis.bin_count())?
                .checked_add(axis.index(*coordinates.get(event)?, final_upper)?)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_edges_and_assigns_boundaries_explicitly() {
        assert!(BinningAxis::new([0.0]).is_err());
        assert!(BinningAxis::new([0.0, f64::NAN]).is_err());
        assert!(BinningAxis::new([0.0, 0.0]).is_err());

        let axis = BinningAxis::new([0.0, 1.0, 3.0]).unwrap();
        assert_eq!(axis.index(0.0, FinalUpperEdge::Exclusive), Some(0));
        assert_eq!(axis.index(1.0, FinalUpperEdge::Exclusive), Some(1));
        assert_eq!(axis.index(3.0, FinalUpperEdge::Exclusive), None);
        assert_eq!(axis.index(3.0, FinalUpperEdge::Inclusive), Some(1));
        assert_eq!(axis.index(f64::NAN, FinalUpperEdge::Inclusive), None);
    }

    #[test]
    fn multidimensional_assignment_is_row_major() {
        let axes = [
            BinningAxis::new([0.0, 1.0, 2.0]).unwrap(),
            BinningAxis::new([10.0, 20.0, 30.0, 40.0]).unwrap(),
        ];
        assert_eq!(bin_shape(&axes), vec![2, 3]);
        assert_eq!(
            flat_bin_index(&axes, &[1.5, 25.0], FinalUpperEdge::Exclusive),
            Some(4)
        );
        assert_eq!(
            flat_bin_index(&axes, &[2.0, 40.0], FinalUpperEdge::Exclusive),
            None
        );
        let columns = vec![vec![1.5], vec![25.0]];
        assert_eq!(
            flat_bin_index_for_event(&axes, &columns, 0, FinalUpperEdge::Exclusive),
            Some(4)
        );
    }
}
