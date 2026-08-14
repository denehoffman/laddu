use fastrand::Rng;
use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::{LadduPhysicsError, LadduPhysicsResult};

/// A simple weighted histogram with explicit bin edges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    /// The number of counts in each bin (can be [`f64`]s since these might be weighted counts)
    counts: Vec<f64>,
    /// The edges of each bin (length is one greater than `counts`)
    bin_edges: Vec<f64>,
    underflow: f64,
    overflow: f64,
    errors: Vec<f64>,
}

#[derive(Clone, Copy)]
enum FillTarget {
    Underflow,
    Bin(usize),
    Overflow,
}

#[derive(Clone, Copy)]
enum TotalWeight {
    InRange,
    WithFlow,
}

#[derive(Clone, Copy)]
struct HistogramRequirements {
    nonnegative: bool,
    positive_total: Option<TotalWeight>,
}

impl HistogramRequirements {
    const VALID: Self = Self {
        nonnegative: false,
        positive_total: None,
    };
    const NORMALIZABLE: Self = Self {
        nonnegative: false,
        positive_total: Some(TotalWeight::InRange),
    };
    const NORMALIZABLE_WITH_FLOW: Self = Self {
        nonnegative: false,
        positive_total: Some(TotalWeight::WithFlow),
    };
    const PROBABILITY_LIKE: Self = Self {
        nonnegative: true,
        positive_total: Some(TotalWeight::InRange),
    };
}

#[derive(Clone, Copy)]
struct TransformPolicy {
    divide_by_bin_width: bool,
    preserve_flow: bool,
}

impl Histogram {
    /// Construct and validate a histogram from weighted bin counts and bin edges.
    ///
    /// The argument order matches `numpy.histogram`, so its result can be forwarded directly.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when counts or edges are non-finite, the
    /// edge count is inconsistent with the bins, or edges are not increasing.
    pub fn new(counts: Vec<f64>, bin_edges: Vec<f64>) -> LadduPhysicsResult<Self> {
        Self::new_with_flow(counts, bin_edges, 0.0, 0.0)
    }

    /// Construct a histogram including explicit underflow and overflow weights.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when counts, flow weights, or edges are
    /// non-finite, lengths are inconsistent, or edges are not increasing.
    pub fn new_with_flow(
        counts: Vec<f64>,
        bin_edges: Vec<f64>,
        underflow: f64,
        overflow: f64,
    ) -> LadduPhysicsResult<Self> {
        let histogram = Self {
            counts: counts.clone(),
            bin_edges,
            underflow,
            overflow,
            errors: counts.into_iter().map(|count| count.abs().sqrt()).collect(),
        };
        histogram.validate_requirements(HistogramRequirements::VALID)?;
        Ok(histogram)
    }

    /// Construct an empty, uniformly binned histogram.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `bins` is zero or the limits are
    /// non-finite or not increasing.
    pub fn empty(bins: usize, limits: (f64, f64)) -> LadduPhysicsResult<Self> {
        Self::validate_bins(bins)?;
        Self::validate_limits(limits)?;
        let bin_edges = Self::calculate_bin_edges(bins, limits);
        Self::empty_with_edges(bin_edges)
    }

    /// Construct an empty histogram from explicit bin edges.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when fewer than two finite, strictly
    /// increasing edges are supplied.
    pub fn empty_with_edges(bin_edges: Vec<f64>) -> LadduPhysicsResult<Self> {
        let counts = vec![0.0; bin_edges.len().saturating_sub(1)];
        Self::new(counts, bin_edges)
    }

    /// Fill a uniformly binned histogram from values and optional weights.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when histogram geometry is invalid,
    /// weights have the wrong length, or a value or weight is non-finite.
    pub fn from_values(
        values: &[f64],
        bins: usize,
        limits: (f64, f64),
        weights: Option<&[f64]>,
    ) -> LadduPhysicsResult<Self> {
        Self::validate_value_weights(values, weights)?;
        let histogram = Self::empty(bins, limits)?;
        Self::fill_values(histogram, values, weights)
    }

    /// Fill an explicitly binned histogram from values and optional weights.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when edges are invalid, weights have the
    /// wrong length, or a value or weight is non-finite.
    pub fn from_values_with_edges(
        values: &[f64],
        bin_edges: Vec<f64>,
        weights: Option<&[f64]>,
    ) -> LadduPhysicsResult<Self> {
        Self::validate_value_weights(values, weights)?;
        let histogram = Self::empty_with_edges(bin_edges)?;
        Self::fill_values(histogram, values, weights)
    }

    fn validate_value_weights(values: &[f64], weights: Option<&[f64]>) -> LadduPhysicsResult<()> {
        if let Some(weights) = weights
            && values.len() != weights.len()
        {
            return Err(LadduPhysicsError::invalid_length(
                "`weights`",
                format!("same length as `values` ({})", values.len()),
                weights.len(),
            ));
        }
        Ok(())
    }

    fn fill_values(
        mut histogram: Self,
        values: &[f64],
        weights: Option<&[f64]>,
    ) -> LadduPhysicsResult<Self> {
        for (i, &value) in values.iter().enumerate() {
            let weight = weights.map_or(1.0, |weights| weights[i]);
            histogram.fill_weighted(value, weight)?;
        }

        Ok(histogram)
    }

    /// Replace the uncertainties on all bins.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `errors` has the wrong length or
    /// contains a negative or non-finite uncertainty.
    pub fn set_errors(&mut self, errors: &[f64]) -> LadduPhysicsResult<()> {
        if self.counts.len() != errors.len() {
            return Err(LadduPhysicsError::invalid_length(
                "`errors`",
                format!("same length as `counts` ({})", self.counts.len(),),
                errors.len(),
            ));
        }

        Self::validate_errors(errors)?;
        self.errors = errors.to_vec();
        Ok(())
    }

    /// Add one unit-weight entry.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `value` is non-finite.
    pub fn fill(&mut self, value: f64) -> LadduPhysicsResult<()> {
        self.fill_weighted(value, 1.0)
    }

    /// Add an entry with an explicit weight.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `value` or `weight` is non-finite.
    ///
    /// # Panics
    ///
    /// Panics if this histogram's validated edge list is unexpectedly empty.
    pub fn fill_weighted(&mut self, value: f64, weight: f64) -> LadduPhysicsResult<()> {
        Self::validate_fill(value, weight)?;
        self.apply_fill(value, weight, weight);
        Ok(())
    }

    /// Add one unit-weight entry with the given entry uncertainty.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `value` is non-finite or `error` is
    /// negative or non-finite.
    pub fn fill_with_error(&mut self, value: f64, error: f64) -> LadduPhysicsResult<()> {
        self.fill_weighted_with_error(value, 1.0, error)
    }

    /// Add an entry with an explicit weight and uncertainty.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `value` or `weight` is non-finite, or
    /// `error` is negative or non-finite.
    ///
    /// # Panics
    ///
    /// Panics if this histogram's validated edge list is unexpectedly empty.
    pub fn fill_weighted_with_error(
        &mut self,
        value: f64,
        weight: f64,
        error: f64,
    ) -> LadduPhysicsResult<()> {
        Self::validate_fill(value, weight)?;
        Self::validate_error("histogram fill error", error)?;
        self.apply_fill(value, weight, error);
        Ok(())
    }

    fn validate_fill(value: f64, weight: f64) -> LadduPhysicsResult<()> {
        if !value.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram fill value",
                "finite",
                value,
            ));
        }

        if !weight.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram fill weight",
                "finite",
                weight,
            ));
        }

        Ok(())
    }

    fn apply_fill(&mut self, value: f64, weight: f64, uncertainty: f64) {
        match self.fill_target(value) {
            Some(FillTarget::Underflow) => self.underflow += weight,
            Some(FillTarget::Bin(index)) => {
                self.counts[index] += weight;
                self.errors[index] = self.errors[index].hypot(uncertainty);
            }
            Some(FillTarget::Overflow) => self.overflow += weight,
            None => {}
        }
    }

    fn fill_target(&self, value: f64) -> Option<FillTarget> {
        if value < self.bin_edges[0] {
            Some(FillTarget::Underflow)
        } else if value >= self.bin_edges[self.bin_edges.len() - 1] {
            Some(FillTarget::Overflow)
        } else {
            self.bin_index(value).map(FillTarget::Bin)
        }
    }

    fn calculate_bin_edges(bins: usize, limits: (f64, f64)) -> Vec<f64> {
        let bin_width = (limits.1 - limits.0) / (bins as f64);
        (0..=bins)
            .map(|i| limits.0 + (i as f64 * bin_width))
            .collect()
    }

    /// Return the number of weighted counts in each bin.
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }

    /// Replace the contents of all bins without changing their uncertainties.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `counts` has the wrong length or
    /// contains a non-finite value.
    pub fn set_counts(&mut self, counts: &[f64]) -> LadduPhysicsResult<()> {
        if self.counts.len() != counts.len() {
            return Err(LadduPhysicsError::invalid_length(
                "`counts`",
                format!("same length as existing `counts` ({})", self.counts.len()),
                counts.len(),
            ));
        }

        Self::validate_counts(counts)?;
        self.counts.copy_from_slice(counts);
        Ok(())
    }

    /// Manually set the counts in a bin.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `bin_index` is out of range or
    /// `value` is non-finite.
    pub fn set_count(&mut self, bin_index: usize, value: f64) -> LadduPhysicsResult<()> {
        if !value.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram bin count",
                "finite",
                value,
            ));
        }

        if bin_index >= self.counts.len() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram bin index",
                format!("less than {}", self.counts.len()),
                bin_index,
            ));
        }

        self.counts[bin_index] = value;
        Ok(())
    }

    /// Manually set the uncertainty in a bin.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when `bin_index` is out of range or
    /// `error` is negative or non-finite.
    pub fn set_error(&mut self, bin_index: usize, error: f64) -> LadduPhysicsResult<()> {
        Self::validate_error("histogram bin error", error)?;

        if bin_index >= self.errors.len() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram bin index",
                format!("less than {}", self.errors.len()),
                bin_index,
            ));
        }

        self.errors[bin_index] = error;
        Ok(())
    }

    /// Return the uncertainties on each bin.
    ///
    /// # Note
    ///
    /// Histograms filled from values use the square root of the sum of squared
    /// weights. Histograms constructed from counts default to
    /// `sqrt(abs(count))`.
    pub fn errors(&self) -> &[f64] {
        &self.errors
    }

    /// Return the bin edges.
    pub fn bin_edges(&self) -> &[f64] {
        &self.bin_edges
    }

    /// Return the accumulated underflow weight.
    pub fn underflow(&self) -> f64 {
        self.underflow
    }

    /// Return the accumulated overflow weight.
    pub fn overflow(&self) -> f64 {
        self.overflow
    }

    /// Return the total histogram weight.
    pub fn total_weight(&self) -> f64 {
        self.counts.iter().sum()
    }

    /// Return total weight including underflow and overflow.
    pub fn total_weight_with_flow(&self) -> f64 {
        self.underflow + self.total_weight() + self.overflow
    }

    /// Return the lowest and highest bin edges.
    pub fn limits(&self) -> (f64, f64) {
        (self.bin_edges[0], self.bin_edges[self.bin_edges.len() - 1])
    }

    /// Return the number of bins.
    pub fn bins(&self) -> usize {
        self.counts.len()
    }

    /// Return the bin index for a value.
    ///
    /// The lower edge is inclusive and the upper edge is exclusive.
    pub fn bin_index(&self, value: f64) -> Option<usize> {
        let (&first, remaining) = self.bin_edges.split_first()?;
        let &last = remaining.last()?;
        if !value.is_finite() {
            return None;
        }

        if value < first || value >= last {
            return None;
        }

        match self
            .bin_edges
            .binary_search_by(|edge| edge.total_cmp(&value))
        {
            Ok(index) => {
                if index == self.counts.len() {
                    None
                } else {
                    Some(index)
                }
            }
            Err(index) => Some(index - 1),
        }
    }

    /// Return a normalized histogram whose in-range bin counts sum to 1.
    ///
    /// Underflow and overflow are discarded because they are outside the
    /// histogram domain.
    ///
    /// Negative bin counts are allowed, so this is an algebraic normalization,
    /// not necessarily a probability distribution.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the histogram is invalid or has zero
    /// or non-finite in-range total weight.
    pub fn normalized(&self) -> LadduPhysicsResult<Self> {
        self.validate_requirements(HistogramRequirements::NORMALIZABLE)?;
        let total_weight = self.total_weight();
        self.transformed(
            TransformPolicy {
                divide_by_bin_width: false,
                preserve_flow: false,
            },
            total_weight,
        )
    }

    /// Return a normalized histogram whose bins plus underflow/overflow sum to 1.
    ///
    /// Negative counts are allowed, so this is algebraic normalization, not
    /// necessarily a probability distribution.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the histogram is invalid or has zero
    /// or non-finite total weight including flow bins.
    pub fn normalized_with_flow(&self) -> LadduPhysicsResult<Self> {
        self.validate_requirements(HistogramRequirements::NORMALIZABLE_WITH_FLOW)?;
        let total_weight = self.total_weight_with_flow();
        self.transformed(
            TransformPolicy {
                divide_by_bin_width: false,
                preserve_flow: true,
            },
            total_weight,
        )
    }

    /// Return a probability density histogram.
    ///
    /// Requires nonnegative in-range counts. Underflow and overflow are discarded
    /// because they do not have finite bin widths.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when bin counts are negative or non-finite,
    /// or their in-range total is not positive and finite.
    pub fn density(&self) -> LadduPhysicsResult<Self> {
        self.validate_requirements(HistogramRequirements::PROBABILITY_LIKE)?;
        let total_weight = self.total_weight();
        self.transformed(
            TransformPolicy {
                divide_by_bin_width: true,
                preserve_flow: false,
            },
            total_weight,
        )
    }

    /// Return a signed density histogram.
    ///
    /// This allows negative weights and is useful for weighted MC, interference
    /// terms, or background-subtracted histograms. It should not be sampled from.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the histogram is invalid or has zero
    /// or non-finite in-range total weight.
    pub fn signed_density(&self) -> LadduPhysicsResult<Self> {
        self.validate_requirements(HistogramRequirements::NORMALIZABLE)?;
        let total_weight = self.total_weight();
        self.transformed(
            TransformPolicy {
                divide_by_bin_width: true,
                preserve_flow: false,
            },
            total_weight,
        )
    }

    fn transformed(&self, policy: TransformPolicy, total_weight: f64) -> LadduPhysicsResult<Self> {
        let scale = |index: usize| {
            if policy.divide_by_bin_width {
                total_weight * (self.bin_edges[index + 1] - self.bin_edges[index])
            } else {
                total_weight
            }
        };
        let counts = self
            .counts
            .iter()
            .enumerate()
            .map(|(index, count)| count / scale(index))
            .collect();
        let errors = self
            .errors
            .iter()
            .enumerate()
            .map(|(index, error)| error / scale(index).abs())
            .collect::<Vec<_>>();
        let (underflow, overflow) = if policy.preserve_flow {
            (self.underflow / total_weight, self.overflow / total_weight)
        } else {
            (0.0, 0.0)
        };

        let mut histogram =
            Self::new_with_flow(counts, self.bin_edges.clone(), underflow, overflow)?;
        histogram.set_errors(&errors)?;
        Ok(histogram)
    }

    /// Sample a value from the histogram, assuming counts define bin probabilities.
    ///
    /// Samples uniformly within the selected bin.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when counts are negative or non-finite, or
    /// their total is not positive and finite.
    pub fn sample(&self, rng: &mut Rng) -> LadduPhysicsResult<f64> {
        self.validate_requirements(HistogramRequirements::PROBABILITY_LIKE)?;

        let total_weight = self.total_weight();
        let mut threshold = rng.f64() * total_weight;

        for (i, count) in self.counts.iter().enumerate() {
            threshold -= count;

            if threshold <= 0.0 {
                let low = self.bin_edges[i];
                let high = self.bin_edges[i + 1];
                return Ok(rng.f64_range(low..high));
            }
        }

        // Handles tiny floating-point roundoff.
        let last = self.counts.len() - 1;
        Ok(rng.f64_range(self.bin_edges[last]..self.bin_edges[last + 1]))
    }

    /// Return the center of a bin.
    pub fn bin_center(&self, index: usize) -> Option<f64> {
        if index < self.counts.len() {
            Some(self.bin_center_unchecked(index))
        } else {
            None
        }
    }

    fn bin_center_unchecked(&self, index: usize) -> f64 {
        0.5 * (self.bin_edges[index] + self.bin_edges[index + 1])
    }

    fn validate_bins(bins: usize) -> LadduPhysicsResult<()> {
        if bins == 0 {
            return Err(LadduPhysicsError::invalid_length(
                "histogram bins",
                "at least 1",
                bins,
            ));
        }
        Ok(())
    }

    fn validate_limits(limits: (f64, f64)) -> LadduPhysicsResult<()> {
        if !limits.0.is_finite() || !limits.1.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram limits",
                "finite lower and upper edges",
                format!("({}, {})", limits.0, limits.1),
            ));
        }

        if limits.1 <= limits.0 {
            return Err(LadduPhysicsError::invalid_relation(format!(
                "histogram upper edge must be greater than lower edge, got ({}, {})",
                limits.0, limits.1
            )));
        }
        Ok(())
    }

    fn validate_structure(&self) -> LadduPhysicsResult<()> {
        if self.bin_edges.len() < 2 {
            return Err(LadduPhysicsError::invalid_length(
                "histogram bin edges",
                "at least 2",
                self.bin_edges.len(),
            ));
        }

        if self.counts.len() + 1 != self.bin_edges.len() {
            return Err(LadduPhysicsError::invalid_length(
                "histogram counts/bin_edges",
                "counts.len() + 1 == bin_edges.len()",
                format!(
                    "{} counts and {} edges",
                    self.counts.len(),
                    self.bin_edges.len()
                ),
            ));
        }

        if self.errors.len() != self.counts.len() {
            return Err(LadduPhysicsError::invalid_length(
                "histogram errors",
                format!("same length as counts ({})", self.counts.len()),
                self.errors.len(),
            ));
        }

        for (index, edges) in self.bin_edges.windows(2).enumerate() {
            if edges[1] <= edges[0] {
                return Err(LadduPhysicsError::invalid_relation(format!(
                    "histogram bin edges must be strictly increasing at edge pair {index}"
                )));
            }
        }

        Ok(())
    }

    fn validate_finite(&self) -> LadduPhysicsResult<()> {
        for (index, edge) in self.bin_edges.iter().enumerate() {
            if !edge.is_finite() {
                return Err(LadduPhysicsError::invalid_value(
                    format!("histogram bin edge {index}"),
                    "finite",
                    *edge,
                ));
            }
        }

        for (index, count) in self.counts.iter().enumerate() {
            if !count.is_finite() {
                return Err(LadduPhysicsError::invalid_value(
                    format!("histogram count {index}"),
                    "finite",
                    *count,
                ));
            }
        }

        Self::validate_errors(&self.errors)?;

        if !self.underflow.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram underflow",
                "finite",
                self.underflow,
            ));
        }

        if !self.overflow.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram overflow",
                "finite",
                self.overflow,
            ));
        }

        Ok(())
    }

    fn validate_counts(counts: &[f64]) -> LadduPhysicsResult<()> {
        for (index, count) in counts.iter().enumerate() {
            if !count.is_finite() {
                return Err(LadduPhysicsError::invalid_value(
                    format!("histogram count {index}"),
                    "finite",
                    *count,
                ));
            }
        }
        Ok(())
    }

    fn validate_error(name: impl Into<String>, error: f64) -> LadduPhysicsResult<()> {
        if !error.is_finite() || error < 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                name,
                "finite and nonnegative",
                error,
            ));
        }
        Ok(())
    }

    fn validate_errors(errors: &[f64]) -> LadduPhysicsResult<()> {
        for (index, error) in errors.iter().enumerate() {
            Self::validate_error(format!("histogram error {index}"), *error)?;
        }
        Ok(())
    }

    fn validate_nonnegative_counts(&self) -> LadduPhysicsResult<()> {
        for (index, count) in self.counts.iter().enumerate() {
            if *count < 0.0 {
                return Err(LadduPhysicsError::invalid_value(
                    format!("histogram count {index}"),
                    "nonnegative",
                    *count,
                ));
            }
        }

        if self.underflow < 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "histogram underflow",
                "nonnegative",
                self.underflow,
            ));
        }

        if self.overflow < 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "histogram overflow",
                "nonnegative",
                self.overflow,
            ));
        }

        Ok(())
    }

    fn validate_positive_total_weight(&self) -> LadduPhysicsResult<()> {
        let total_weight = self.total_weight();

        if total_weight <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "histogram total weight",
                "positive",
                total_weight,
            ));
        }

        Ok(())
    }

    fn validate_positive_total_weight_with_flow(&self) -> LadduPhysicsResult<()> {
        let total_weight = self.total_weight_with_flow();

        if total_weight <= 0.0 {
            return Err(LadduPhysicsError::invalid_value(
                "histogram total weight with flow",
                "positive",
                total_weight,
            ));
        }

        Ok(())
    }

    fn validate_requirements(&self, requirements: HistogramRequirements) -> LadduPhysicsResult<()> {
        self.validate_structure()?;
        self.validate_finite()?;
        if requirements.nonnegative {
            self.validate_nonnegative_counts()?;
        }
        match requirements.positive_total {
            Some(TotalWeight::InRange) => self.validate_positive_total_weight()?,
            Some(TotalWeight::WithFlow) => self.validate_positive_total_weight_with_flow()?,
            None => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn new_accepts_valid_histograms() {
        let hist = Histogram::new(vec![2.0], vec![0.0, 1.0]).unwrap();

        assert_relative_eq!(hist.counts(), &[2.0][..]);
        assert_relative_eq!(hist.bin_edges(), &[0.0, 1.0][..]);
        assert_relative_eq!(hist.underflow(), 0.0);
        assert_relative_eq!(hist.overflow(), 0.0);
    }

    #[test]
    fn new_accepts_zero_and_negative_weight_histograms() {
        assert!(Histogram::new(vec![0.0], vec![0.0, 1.0]).is_ok());
        assert!(Histogram::new(vec![-1.0], vec![0.0, 1.0]).is_ok());
    }

    #[test]
    fn new_with_flow_accepts_valid_flow() {
        let hist = Histogram::new_with_flow(vec![2.0], vec![0.0, 1.0], 3.0, 4.0).unwrap();

        assert_relative_eq!(hist.counts(), &[2.0][..]);
        assert_relative_eq!(hist.underflow(), 3.0);
        assert_relative_eq!(hist.overflow(), 4.0);
        assert_relative_eq!(hist.total_weight(), 2.0);
        assert_relative_eq!(hist.total_weight_with_flow(), 9.0);
    }

    #[test]
    fn new_rejects_invalid_structure() {
        assert!(Histogram::new(vec![], vec![0.0]).is_err());
        assert!(Histogram::new(vec![1.0, 2.0], vec![0.0, 1.0]).is_err());
        assert!(Histogram::new(vec![1.0], vec![0.0, 0.0]).is_err());
        assert!(Histogram::new(vec![1.0], vec![0.0, -1.0]).is_err());
    }

    #[test]
    fn new_rejects_nonfinite_values() {
        assert!(Histogram::new(vec![1.0], vec![0.0, f64::NAN]).is_err());
        assert!(Histogram::new(vec![1.0], vec![0.0, f64::INFINITY]).is_err());
        assert!(Histogram::new(vec![f64::NAN], vec![0.0, 1.0]).is_err());
        assert!(Histogram::new(vec![f64::INFINITY], vec![0.0, 1.0]).is_err());

        assert!(Histogram::new_with_flow(vec![1.0], vec![0.0, 1.0], f64::NAN, 0.0).is_err());
        assert!(Histogram::new_with_flow(vec![1.0], vec![0.0, 1.0], 0.0, f64::NAN).is_err());
        assert!(Histogram::new_with_flow(vec![1.0], vec![0.0, 1.0], f64::INFINITY, 0.0).is_err());
        assert!(Histogram::new_with_flow(vec![1.0], vec![0.0, 1.0], 0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn empty_constructs_evenly_spaced_histogram() {
        let hist = Histogram::empty(4, (0.0, 1.0)).unwrap();

        assert_eq!(hist.bins(), 4);
        assert_eq!(hist.limits(), (0.0, 1.0));
        assert_relative_eq!(hist.counts(), &[0.0, 0.0, 0.0, 0.0][..]);
        assert_relative_eq!(hist.bin_edges(), &[0.0, 0.25, 0.5, 0.75, 1.0][..]);
    }

    #[test]
    fn empty_rejects_invalid_bins_and_limits() {
        assert!(Histogram::empty(0, (0.0, 1.0)).is_err());
        assert!(Histogram::empty(4, (1.0, 0.0)).is_err());
        assert!(Histogram::empty(4, (0.0, 0.0)).is_err());
        assert!(Histogram::empty(4, (f64::NAN, 1.0)).is_err());
        assert!(Histogram::empty(4, (0.0, f64::NAN)).is_err());
    }

    #[test]
    fn empty_with_edges_constructs_nonuniform_empty_histogram() {
        let hist = Histogram::empty_with_edges(vec![0.0, 0.1, 0.4, 1.0]).unwrap();

        assert_eq!(hist.bins(), 3);
        assert_eq!(hist.limits(), (0.0, 1.0));
        assert_relative_eq!(hist.counts(), &[0.0, 0.0, 0.0][..]);
        assert_relative_eq!(hist.bin_edges(), &[0.0, 0.1, 0.4, 1.0][..]);
    }

    #[test]
    fn from_values_fills_even_histogram_without_weights() {
        let values = vec![-0.1, 0.0, 0.2, 0.25, 0.7, 0.99, 1.0, 1.2];
        let hist = Histogram::from_values(&values, 4, (0.0, 1.0), None).unwrap();

        assert_relative_eq!(hist.counts(), &[2.0, 1.0, 1.0, 1.0][..]);
        assert_relative_eq!(hist.underflow(), 1.0);
        assert_relative_eq!(hist.overflow(), 2.0);
        assert_relative_eq!(hist.total_weight(), 5.0);
        assert_relative_eq!(hist.total_weight_with_flow(), 8.0);
    }

    #[test]
    fn from_values_fills_even_histogram_with_weights() {
        let values = vec![-0.1, 0.1, 0.4, 0.8, 1.2];
        let weights = vec![10.0, 1.0, 2.0, 3.0, 20.0];

        let hist = Histogram::from_values(&values, 2, (0.0, 1.0), Some(&weights)).unwrap();

        assert_relative_eq!(hist.counts(), &[3.0, 3.0][..]);
        assert_relative_eq!(hist.underflow(), 10.0);
        assert_relative_eq!(hist.overflow(), 20.0);
        assert_relative_eq!(hist.total_weight(), 6.0);
        assert_relative_eq!(hist.total_weight_with_flow(), 36.0);
    }

    #[test]
    fn from_values_rejects_mismatched_weights() {
        let values = vec![0.1, 0.2];
        let weights = vec![1.0];

        assert!(Histogram::from_values(&values, 2, (0.0, 1.0), Some(&weights)).is_err());
    }

    #[test]
    fn from_values_rejects_nonfinite_values_and_weights() {
        assert!(Histogram::from_values(&[f64::NAN], 2, (0.0, 1.0), None).is_err());
        assert!(Histogram::from_values(&[0.5], 2, (0.0, 1.0), Some(&[f64::NAN])).is_err());
        assert!(Histogram::from_values(&[0.5], 2, (0.0, 1.0), Some(&[f64::INFINITY])).is_err());
    }

    #[test]
    fn from_values_with_edges_fills_nonuniform_histogram() {
        let values = vec![-0.1, 0.0, 0.05, 0.1, 0.39, 0.4, 0.99, 1.0];
        let hist =
            Histogram::from_values_with_edges(&values, vec![0.0, 0.1, 0.4, 1.0], None).unwrap();

        assert_relative_eq!(hist.counts(), &[2.0, 2.0, 2.0][..]);
        assert_relative_eq!(hist.underflow(), 1.0);
        assert_relative_eq!(hist.overflow(), 1.0);
    }

    #[test]
    fn from_values_with_edges_rejects_mismatched_weights() {
        let values = vec![0.1, 0.2];
        let weights = vec![1.0];

        assert!(
            Histogram::from_values_with_edges(&values, vec![0.0, 1.0], Some(&weights)).is_err()
        );
    }

    #[test]
    fn fill_adds_unit_weight() {
        let mut hist = Histogram::empty(2, (0.0, 1.0)).unwrap();

        hist.fill(0.25).unwrap();
        hist.fill(0.75).unwrap();
        hist.fill(0.75).unwrap();

        assert_relative_eq!(hist.counts(), &[1.0, 2.0][..]);
    }

    #[test]
    fn fill_weighted_tracks_underflow_and_overflow() {
        let mut hist = Histogram::empty(2, (0.0, 1.0)).unwrap();

        hist.fill_weighted(-0.1, 2.0).unwrap();
        hist.fill_weighted(0.0, 3.0).unwrap();
        hist.fill_weighted(0.5, 4.0).unwrap();
        hist.fill_weighted(1.0, 5.0).unwrap();

        assert_relative_eq!(hist.counts(), &[3.0, 4.0][..]);
        assert_relative_eq!(hist.underflow(), 2.0);
        assert_relative_eq!(hist.overflow(), 5.0);
    }

    #[test]
    fn fill_paths_share_boundary_routing_and_uncertainty_policy() {
        let cases = [
            (-0.1, FillTarget::Underflow),
            (0.0, FillTarget::Bin(0)),
            (0.5, FillTarget::Bin(1)),
            (1.0, FillTarget::Overflow),
        ];

        for (value, target) in cases {
            let mut default_error = Histogram::empty(2, (0.0, 1.0)).unwrap();
            let mut explicit_error = default_error.clone();

            default_error.fill_weighted(value, -2.0).unwrap();
            explicit_error
                .fill_weighted_with_error(value, -2.0, 3.0)
                .unwrap();

            match target {
                FillTarget::Underflow => {
                    assert_relative_eq!(default_error.underflow(), -2.0);
                    assert_relative_eq!(explicit_error.underflow(), -2.0);
                }
                FillTarget::Bin(index) => {
                    assert_relative_eq!(default_error.counts()[index], -2.0);
                    assert_relative_eq!(default_error.errors()[index], 2.0);
                    assert_relative_eq!(explicit_error.counts()[index], -2.0);
                    assert_relative_eq!(explicit_error.errors()[index], 3.0);
                }
                FillTarget::Overflow => {
                    assert_relative_eq!(default_error.overflow(), -2.0);
                    assert_relative_eq!(explicit_error.overflow(), -2.0);
                }
            }
        }
    }

    #[test]
    fn fill_weighted_accepts_negative_weights() {
        let mut hist = Histogram::empty(2, (0.0, 1.0)).unwrap();

        hist.fill_weighted(0.25, -2.0).unwrap();
        hist.fill_weighted(-0.1, -3.0).unwrap();
        hist.fill_weighted(1.0, -4.0).unwrap();

        assert_relative_eq!(hist.counts(), &[-2.0, 0.0][..]);
        assert_relative_eq!(hist.underflow(), -3.0);
        assert_relative_eq!(hist.overflow(), -4.0);
    }

    #[test]
    fn fill_weighted_rejects_nonfinite_value_or_weight() {
        let mut hist = Histogram::empty(2, (0.0, 1.0)).unwrap();

        assert!(hist.fill_weighted(f64::NAN, 1.0).is_err());
        assert!(hist.fill_weighted(f64::INFINITY, 1.0).is_err());
        assert!(hist.fill_weighted(0.5, f64::NAN).is_err());
        assert!(hist.fill_weighted(0.5, f64::INFINITY).is_err());
    }

    #[test]
    fn errors_default_to_sqrt_absolute_counts() {
        let hist = Histogram::new(vec![4.0, -9.0], vec![0.0, 1.0, 2.0]).unwrap();

        assert_relative_eq!(hist.errors(), &[2.0, 3.0][..]);
    }

    #[test]
    fn weighted_fills_accumulate_uncertainties_in_quadrature() {
        let mut hist = Histogram::empty(1, (0.0, 1.0)).unwrap();

        hist.fill_weighted(0.5, 3.0).unwrap();
        hist.fill_weighted(0.5, -4.0).unwrap();

        assert_relative_eq!(hist.counts(), &[-1.0][..]);
        assert_relative_eq!(hist.errors(), &[5.0][..]);
    }

    #[test]
    fn explicit_fill_errors_accumulate_in_quadrature() {
        let mut hist = Histogram::empty(1, (0.0, 1.0)).unwrap();

        hist.fill_weighted_with_error(0.5, 10.0, 3.0).unwrap();
        hist.fill_weighted_with_error(0.5, 20.0, 4.0).unwrap();

        assert_relative_eq!(hist.counts(), &[30.0][..]);
        assert_relative_eq!(hist.errors(), &[5.0][..]);
        assert!(hist.fill_with_error(0.5, -1.0).is_err());
    }

    #[test]
    fn manual_counts_and_errors_are_validated() {
        let mut hist = Histogram::empty(2, (0.0, 1.0)).unwrap();

        hist.set_counts(&[2.0, -3.0]).unwrap();
        hist.set_errors(&[0.5, 1.5]).unwrap();
        hist.set_count(1, 4.0).unwrap();
        hist.set_error(0, 0.25).unwrap();

        assert_relative_eq!(hist.counts(), &[2.0, 4.0][..]);
        assert_relative_eq!(hist.errors(), &[0.25, 1.5][..]);
        assert!(hist.set_counts(&[1.0]).is_err());
        assert!(hist.set_counts(&[1.0, f64::NAN]).is_err());
        assert!(hist.set_errors(&[1.0]).is_err());
        assert!(hist.set_errors(&[1.0, -1.0]).is_err());
        assert!(hist.set_count(2, 1.0).is_err());
        assert!(hist.set_error(2, 1.0).is_err());
    }

    #[test]
    fn bin_index_uses_lower_inclusive_upper_exclusive_edges() {
        let hist = Histogram::empty(4, (0.0, 1.0)).unwrap();

        assert_eq!(hist.bin_index(-0.1), None);
        assert_eq!(hist.bin_index(0.0), Some(0));
        assert_eq!(hist.bin_index(0.249), Some(0));
        assert_eq!(hist.bin_index(0.25), Some(1));
        assert_eq!(hist.bin_index(0.5), Some(2));
        assert_eq!(hist.bin_index(0.75), Some(3));
        assert_eq!(hist.bin_index(0.999), Some(3));
        assert_eq!(hist.bin_index(1.0), None);
    }

    #[test]
    fn bin_index_handles_nonuniform_edges() {
        let hist = Histogram::empty_with_edges(vec![0.0, 0.1, 0.4, 1.0]).unwrap();

        assert_eq!(hist.bin_index(0.0), Some(0));
        assert_eq!(hist.bin_index(0.099), Some(0));
        assert_eq!(hist.bin_index(0.1), Some(1));
        assert_eq!(hist.bin_index(0.399), Some(1));
        assert_eq!(hist.bin_index(0.4), Some(2));
        assert_eq!(hist.bin_index(0.999), Some(2));
        assert_eq!(hist.bin_index(1.0), None);
    }

    #[test]
    fn normalized_scales_counts_by_in_range_weight() {
        let hist = Histogram::new_with_flow(vec![2.0, 6.0], vec![0.0, 1.0, 2.0], 4.0, 8.0).unwrap();

        let normalized = hist.normalized().unwrap();

        assert_relative_eq!(normalized.counts(), &[0.25, 0.75][..]);
        assert_relative_eq!(normalized.underflow(), 0.0);
        assert_relative_eq!(normalized.overflow(), 0.0);
        assert_relative_eq!(normalized.total_weight(), 1.0);
        assert_relative_eq!(normalized.total_weight_with_flow(), 1.0);
    }

    #[test]
    fn normalization_and_density_scale_errors() {
        let mut hist = Histogram::new(vec![2.0, 6.0], vec![0.0, 1.0, 3.0]).unwrap();
        hist.set_errors(&[1.0, 3.0]).unwrap();

        let normalized = hist.normalized().unwrap();
        assert_relative_eq!(normalized.errors(), &[0.125, 0.375][..]);

        let density = hist.density().unwrap();
        assert_relative_eq!(density.errors(), &[0.125, 0.1875][..]);
    }

    #[test]
    fn normalized_with_flow_scales_counts_and_flow_by_total_weight_with_flow() {
        let hist = Histogram::new_with_flow(vec![2.0, 6.0], vec![0.0, 1.0, 2.0], 4.0, 8.0).unwrap();

        let normalized = hist.normalized_with_flow().unwrap();

        assert_relative_eq!(normalized.counts(), &[0.1, 0.3][..]);
        assert_relative_eq!(normalized.underflow(), 0.2);
        assert_relative_eq!(normalized.overflow(), 0.4);
        assert_relative_eq!(normalized.total_weight(), 0.4);
        assert_relative_eq!(normalized.total_weight_with_flow(), 1.0);
    }

    #[test]
    fn normalized_rejects_zero_in_range_weight() {
        let hist = Histogram::new_with_flow(vec![0.0], vec![0.0, 1.0], 1.0, 1.0).unwrap();

        assert!(hist.normalized().is_err());
    }

    #[test]
    fn density_converts_counts_to_probability_density_and_drops_flow() {
        let hist = Histogram::new_with_flow(vec![2.0, 6.0], vec![0.0, 1.0, 3.0], 4.0, 8.0).unwrap();

        let density = hist.density().unwrap();

        assert_relative_eq!(density.counts(), &[0.25, 0.375][..]);
        assert_relative_eq!(density.underflow(), 0.0);
        assert_relative_eq!(density.overflow(), 0.0);

        let integral = density.counts()[0] * 1.0 + density.counts()[1] * 2.0;
        assert_relative_eq!(integral, 1.0);
    }

    #[test]
    fn transforms_match_reference_formulas_for_uniform_and_nonuniform_bins() {
        let mut rng = Rng::with_seed(0x5eed);

        for uniform in [true, false] {
            for _ in 0..64 {
                let mut edges = vec![rng.f64_range(-5.0..0.0)];
                let uniform_width = rng.f64_range(0.1..2.0);
                for _ in 0..4 {
                    let width = if uniform {
                        uniform_width
                    } else {
                        rng.f64_range(0.1..2.0)
                    };
                    edges.push(edges.last().unwrap() + width);
                }

                let counts = (0..4).map(|_| rng.f64_range(0.1..10.0)).collect::<Vec<_>>();
                let errors = (0..4).map(|_| rng.f64_range(0.0..3.0)).collect::<Vec<_>>();
                let underflow = rng.f64_range(0.0..3.0);
                let overflow = rng.f64_range(0.0..3.0);
                let mut hist =
                    Histogram::new_with_flow(counts, edges.clone(), underflow, overflow).unwrap();
                hist.set_errors(&errors).unwrap();

                let in_range_total = hist.total_weight();
                let total_with_flow = hist.total_weight_with_flow();
                let normalized = hist.normalized().unwrap();
                let normalized_with_flow = hist.normalized_with_flow().unwrap();
                let density = hist.density().unwrap();
                let signed_density = hist.signed_density().unwrap();

                for index in 0..hist.bins() {
                    let width = edges[index + 1] - edges[index];
                    assert_relative_eq!(
                        normalized.counts()[index],
                        hist.counts()[index] / in_range_total
                    );
                    assert_relative_eq!(
                        normalized.errors()[index],
                        hist.errors()[index] / in_range_total
                    );
                    assert_relative_eq!(
                        normalized_with_flow.counts()[index],
                        hist.counts()[index] / total_with_flow
                    );
                    assert_relative_eq!(
                        normalized_with_flow.errors()[index],
                        hist.errors()[index] / total_with_flow
                    );
                    assert_relative_eq!(
                        density.counts()[index],
                        hist.counts()[index] / (in_range_total * width)
                    );
                    assert_relative_eq!(
                        density.errors()[index],
                        hist.errors()[index] / (in_range_total * width)
                    );
                    assert_relative_eq!(signed_density.counts()[index], density.counts()[index]);
                    assert_relative_eq!(signed_density.errors()[index], density.errors()[index]);
                }

                assert_relative_eq!(
                    normalized_with_flow.underflow(),
                    underflow / total_with_flow
                );
                assert_relative_eq!(normalized_with_flow.overflow(), overflow / total_with_flow);
                assert_relative_eq!(normalized.underflow(), 0.0);
                assert_relative_eq!(density.overflow(), 0.0);
            }
        }
    }

    #[test]
    fn density_rejects_negative_counts_or_flow() {
        let negative_count = Histogram::new(vec![-1.0], vec![0.0, 1.0]).unwrap();
        assert!(negative_count.density().is_err());

        let negative_underflow =
            Histogram::new_with_flow(vec![1.0], vec![0.0, 1.0], -1.0, 0.0).unwrap();
        assert!(negative_underflow.density().is_err());

        let negative_overflow =
            Histogram::new_with_flow(vec![1.0], vec![0.0, 1.0], 0.0, -1.0).unwrap();
        assert!(negative_overflow.density().is_err());
    }

    #[test]
    fn sample_returns_value_inside_histogram_limits() {
        let hist = Histogram::new(vec![1.0, 1.0], vec![0.0, 1.0, 2.0]).unwrap();
        let mut rng = Rng::with_seed(12345);

        for _ in 0..100 {
            let value = hist.sample(&mut rng).unwrap();
            assert!((0.0..2.0).contains(&value));
        }
    }

    #[test]
    fn sample_rejects_non_probability_like_histograms() {
        let negative_count = Histogram::new(vec![-1.0], vec![0.0, 1.0]).unwrap();
        assert!(negative_count.sample(&mut Rng::with_seed(1)).is_err());

        let zero_count = Histogram::new(vec![0.0], vec![0.0, 1.0]).unwrap();
        assert!(zero_count.sample(&mut Rng::with_seed(1)).is_err());
    }

    #[test]
    fn bin_center_returns_center_for_valid_index() {
        let hist = Histogram::empty_with_edges(vec![0.0, 0.5, 2.0]).unwrap();

        assert_relative_eq!(hist.bin_center(0).unwrap(), 0.25);
        assert_relative_eq!(hist.bin_center(1).unwrap(), 1.25);
    }

    #[test]
    fn bin_center_returns_none_for_invalid_index() {
        let hist = Histogram::empty_with_edges(vec![0.0, 0.5, 2.0]).unwrap();

        assert_eq!(hist.bin_center(2), None);
    }
}
