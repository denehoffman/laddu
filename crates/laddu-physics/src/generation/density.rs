use std::sync::Arc;

use super::ProposalRng;
use crate::{LadduPhysicsError, LadduPhysicsResult, histogram::Histogram};

#[derive(Clone, Copy, Debug)]
pub(super) struct FiniteInterval {
    low: f64,
    high: f64,
}

impl FiniteInterval {
    pub(super) fn new(low: f64, high: f64) -> Option<Self> {
        (low.is_finite() && high.is_finite() && high > low).then_some(Self { low, high })
    }

    pub(super) fn intersect(self, other: Self) -> Option<Self> {
        Self::new(self.low.max(other.low), self.high.min(other.high))
    }

    pub(super) fn low(self) -> f64 {
        self.low
    }

    pub(super) fn high(self) -> f64 {
        self.high
    }

    pub(super) fn width(self) -> f64 {
        self.high - self.low
    }

    pub(super) fn contains(self, value: f64) -> bool {
        value >= self.low && value <= self.high
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Sample {
    pub(super) value: f64,
    pub(super) inverse_density: f64,
}

/// Validated piecewise-constant one-dimensional density.
///
/// This type is public only to share proposal mechanics with workspace
/// generators. It is not part of the stable user-facing proposal API.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct PiecewiseDensity {
    edges: Arc<[f64]>,
    counts: Arc<[f64]>,
    uniform: Option<(f64, f64, f64)>,
}

impl PiecewiseDensity {
    /// Construct a density over explicit bin edges and nonnegative bin counts.
    #[doc(hidden)]
    pub fn new(edges: Arc<[f64]>, counts: Arc<[f64]>) -> LadduPhysicsResult<Self> {
        if counts.is_empty()
            || edges.len() != counts.len() + 1
            || edges.iter().any(|edge| !edge.is_finite())
            || edges.windows(2).any(|pair| pair[1] <= pair[0])
            || counts
                .iter()
                .any(|count| !count.is_finite() || *count < 0.0)
            || !counts.iter().sum::<f64>().is_finite()
            || counts.iter().sum::<f64>() <= 0.0
        {
            return Err(LadduPhysicsError::invalid_relation(
                "piecewise density requires finite increasing edges and nonnegative finite counts with positive total",
            ));
        }
        Ok(Self {
            edges,
            counts,
            uniform: None,
        })
    }

    /// Construct equally spaced bins over `[low, high]`.
    #[doc(hidden)]
    pub fn uniform(low: f64, high: f64, counts: Arc<[f64]>) -> LadduPhysicsResult<Self> {
        let Some(interval) = FiniteInterval::new(low, high) else {
            return Err(LadduPhysicsError::invalid_relation(
                "piecewise density requires finite bounds with low < high",
            ));
        };
        let width = interval.width() / counts.len() as f64;
        let edges = (0..=counts.len())
            .map(|index| low + index as f64 * width)
            .collect::<Arc<[_]>>();
        let mut density = Self::new(edges, counts)?;
        density.uniform = Some((low, high, width));
        Ok(density)
    }

    pub(super) fn from_histogram(histogram: &Histogram) -> LadduPhysicsResult<Self> {
        Self::new(
            Arc::from(histogram.bin_edges()),
            Arc::from(histogram.counts()),
        )
    }

    pub(super) fn support(&self) -> (f64, f64) {
        let first = self
            .counts
            .iter()
            .position(|count| *count > 0.0)
            .expect("validated density has a positive bin");
        let last = self
            .counts
            .iter()
            .rposition(|count| *count > 0.0)
            .expect("validated density has a positive bin");
        (self.bin_bounds(first).0, self.bin_bounds(last).1)
    }

    fn bin_bounds(&self, index: usize) -> (f64, f64) {
        if let Some((low, _, width)) = self.uniform {
            let bin_low = low + index as f64 * width;
            (bin_low, bin_low + width)
        } else {
            (self.edges[index], self.edges[index + 1])
        }
    }

    fn segments(&self, low: f64, high: f64) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
        let interval = FiniteInterval::new(low, high);
        self.counts
            .iter()
            .enumerate()
            .filter_map(move |(index, count)| {
                let interval = interval?;
                let (bin_low, bin_high) = self.bin_bounds(index);
                let bin = FiniteInterval::new(bin_low, bin_high)?;
                let overlap = interval.intersect(bin)?;
                (*count > 0.0).then_some((
                    overlap.low(),
                    overlap.high(),
                    *count * overlap.width() / bin.width(),
                ))
            })
    }

    /// Return the total bin weight after clipping to an interval.
    #[doc(hidden)]
    pub fn truncated_total(&self, low: f64, high: f64) -> f64 {
        self.segments(low, high).map(|(_, _, weight)| weight).sum()
    }

    /// Evaluate the normalized density after clipping to an interval.
    #[doc(hidden)]
    pub fn density(&self, low: f64, high: f64, value: f64) -> f64 {
        if self
            .uniform
            .is_some_and(|(domain_low, domain_high, _)| value < domain_low || value > domain_high)
        {
            return 0.0;
        }
        let total = self.truncated_total(low, high);
        if total <= 0.0 {
            return 0.0;
        }
        let effective_high = high.min(self.bin_bounds(self.counts.len() - 1).1);
        self.segments(low, high)
            .find(|(segment_low, segment_high, _)| {
                value >= *segment_low
                    && (value < *segment_high
                        || (*segment_high == effective_high && value == effective_high))
            })
            .map_or(0.0, |(segment_low, segment_high, weight)| {
                weight / ((segment_high - segment_low) * total)
            })
    }

    pub(super) fn density_inclusive(&self, low: f64, high: f64, value: f64) -> f64 {
        let total = self.truncated_total(low, high);
        if total <= 0.0 {
            return 0.0;
        }
        self.segments(low, high)
            .find(|(segment_low, segment_high, _)| value >= *segment_low && value <= *segment_high)
            .map_or(0.0, |(segment_low, segment_high, weight)| {
                weight / ((segment_high - segment_low) * total)
            })
    }

    /// Sample using one unit variate for both bin selection and bin position.
    #[doc(hidden)]
    pub fn sample_with_unit(&self, low: f64, high: f64, unit: f64) -> Option<f64> {
        let total = self.truncated_total(low, high);
        let mut threshold = unit * total;
        let mut last_high = None;
        for (segment_low, segment_high, weight) in self.segments(low, high) {
            last_high = Some(segment_high);
            if threshold <= weight {
                return Some(segment_low + threshold / weight * (segment_high - segment_low));
            }
            threshold -= weight;
        }
        last_high
    }

    /// Sample with independent unit variates for bin selection and position.
    #[doc(hidden)]
    pub fn sample(&self, low: f64, high: f64, rng: &mut ProposalRng) -> Option<f64> {
        let total = self.truncated_total(low, high);
        if total <= 0.0 {
            return None;
        }
        let mut threshold = rng.uniform() * total;
        for (segment_low, segment_high, weight) in self.segments(low, high) {
            if threshold <= weight {
                return Some(segment_low + rng.uniform() * (segment_high - segment_low));
            }
            threshold -= weight;
        }
        None
    }

    pub(super) fn sample_indexed(&self, rng: &mut ProposalRng) -> Option<f64> {
        let (low, _, width) = self.uniform?;
        let total: f64 = self.counts.iter().sum();
        let mut threshold = rng.uniform() * total;
        let mut selected = self.counts.len() - 1;
        for (bin, count) in self.counts.iter().enumerate() {
            if threshold <= *count {
                selected = bin;
                break;
            }
            threshold -= count;
        }
        Some(low + (selected as f64 + rng.uniform()) * width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_preserves_density_normalization_and_sample_support() {
        let density = PiecewiseDensity::uniform(0.0, 4.0, Arc::from([1.0, 3.0, 2.0, 4.0])).unwrap();
        let low = 0.5;
        let high = 3.25;
        let steps = 22_000;
        let dx = (high - low) / steps as f64;
        let integral = (0..steps)
            .map(|step| density.density(low, high, low + (step as f64 + 0.5) * dx) * dx)
            .sum::<f64>();
        assert!((integral - 1.0).abs() < 1e-12);

        let mut rng = ProposalRng::new(17);
        for _ in 0..1_000 {
            let value = density.sample(low, high, &mut rng).unwrap();
            assert!((low..=high).contains(&value));
            assert!(density.density(low, high, value).is_finite());
            assert!(density.density(low, high, value) > 0.0);
        }
    }

    #[test]
    fn density_uses_right_bin_at_internal_edges_and_includes_domain_high() {
        let density = PiecewiseDensity::uniform(0.0, 2.0, Arc::from([1.0, 3.0])).unwrap();
        assert_eq!(density.density(-1.0, 3.0, 1.0), 0.75);
        assert_eq!(density.density(-1.0, 3.0, 2.0), 0.75);
        assert_eq!(density.density(-1.0, 3.0, 2.1), 0.0);
    }
}
