use fastrand::Rng;
use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    binning::{BinningAxis, FinalUpperEdge},
};

/// Whether a histogram's stored constituents define reportable uncertainties.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistogramUncertaintyStatus {
    /// The operands were independent, or independence was explicitly asserted.
    #[default]
    Available,
    /// At least one source was shared or unknown, so covariance is unavailable.
    UnavailableCovariance,
}

pub(crate) fn uncertainty_is_available(status: &HistogramUncertaintyStatus) -> bool {
    *status == HistogramUncertaintyStatus::Available
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum HistogramFillKind {
    Manual,
    Empirical,
}

/// A simple weighted histogram with explicit bin edges.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Histogram {
    fill_kind: HistogramFillKind,
    /// The number of counts in each bin (can be [`f64`]s since these might be weighted counts)
    counts: Vec<f64>,
    /// The validated edges of each bin (length is one greater than `counts`).
    #[serde(rename = "bin_edges")]
    axis: BinningAxis,
    underflow: f64,
    overflow: f64,
    errors: Vec<f64>,
    sum_squared_weights: Vec<f64>,
    underflow_sum_squared_weights: Option<f64>,
    overflow_sum_squared_weights: Option<f64>,
    #[serde(default, skip_serializing_if = "uncertainty_is_available")]
    uncertainty_status: HistogramUncertaintyStatus,
    #[serde(skip)]
    corrections: Box<HistogramCorrections>,
}

#[derive(Clone, Debug, PartialEq)]
struct HistogramCorrections {
    counts: Vec<f64>,
    sum_squared_weights: Vec<f64>,
    underflow: f64,
    overflow: f64,
    underflow_sum_squared_weights: f64,
    overflow_sum_squared_weights: f64,
}

impl HistogramCorrections {
    fn new(bins: usize) -> Self {
        Self {
            counts: vec![0.0; bins],
            sum_squared_weights: vec![0.0; bins],
            underflow: 0.0,
            overflow: 0.0,
            underflow_sum_squared_weights: 0.0,
            overflow_sum_squared_weights: 0.0,
        }
    }
}

#[derive(Deserialize)]
struct SerializedHistogram {
    fill_kind: HistogramFillKind,
    counts: Vec<f64>,
    bin_edges: Vec<f64>,
    underflow: f64,
    overflow: f64,
    errors: Vec<f64>,
    sum_squared_weights: Vec<f64>,
    underflow_sum_squared_weights: Option<f64>,
    overflow_sum_squared_weights: Option<f64>,
    #[serde(default)]
    uncertainty_status: HistogramUncertaintyStatus,
}

impl<'de> Deserialize<'de> for Histogram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let SerializedHistogram {
            fill_kind,
            counts,
            bin_edges,
            underflow,
            overflow,
            errors: serialized_errors,
            sum_squared_weights,
            underflow_sum_squared_weights,
            overflow_sum_squared_weights,
            uncertainty_status,
        } = SerializedHistogram::deserialize(deserializer)?;
        let errors = sum_squared_weights
            .iter()
            .map(|sum_squared_weight| sum_squared_weight.sqrt())
            .collect::<Vec<_>>();
        if serialized_errors.len() != errors.len()
            || serialized_errors
                .iter()
                .zip(&errors)
                .any(|(actual, expected)| {
                    !actual.is_finite()
                        || (actual - expected).abs()
                            > f64::EPSILON * 4.0 * actual.abs().max(expected.abs()).max(1.0)
                })
        {
            return Err(serde::de::Error::custom(
                "histogram errors must match the square root of sum_squared_weights",
            ));
        }
        let bins = counts.len();
        let axis = Self::axis_for_edges(&bin_edges).map_err(serde::de::Error::custom)?;
        let histogram = Self {
            fill_kind,
            counts,
            axis,
            underflow,
            overflow,
            errors,
            sum_squared_weights,
            underflow_sum_squared_weights,
            overflow_sum_squared_weights,
            uncertainty_status,
            corrections: Box::new(HistogramCorrections::new(bins)),
        };
        histogram
            .validate_requirements(HistogramRequirements::VALID)
            .map_err(serde::de::Error::custom)?;
        Ok(histogram)
    }
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
    fn axis_for_edges(edges: &[f64]) -> LadduPhysicsResult<BinningAxis> {
        if edges.len() < 2 {
            return Err(LadduPhysicsError::invalid_length(
                "histogram bin edges",
                "at least 2",
                edges.len(),
            ));
        }
        for (index, edge) in edges.iter().enumerate() {
            if !edge.is_finite() {
                return Err(LadduPhysicsError::invalid_value(
                    format!("histogram bin edge {index}"),
                    "finite",
                    *edge,
                ));
            }
        }
        for (index, pair) in edges.windows(2).enumerate() {
            if pair[1] <= pair[0] {
                return Err(LadduPhysicsError::invalid_relation(format!(
                    "histogram bin edges must be strictly increasing at edge pair {index}"
                )));
            }
        }
        BinningAxis::new(edges.iter().copied())
    }

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
        let bins = counts.len();
        let axis = Self::axis_for_edges(&bin_edges)?;
        let histogram = Self {
            fill_kind: HistogramFillKind::Manual,
            counts: counts.clone(),
            axis,
            underflow,
            overflow,
            errors: counts.iter().map(|count| count.abs().sqrt()).collect(),
            sum_squared_weights: counts.into_iter().map(f64::abs).collect(),
            underflow_sum_squared_weights: None,
            overflow_sum_squared_weights: None,
            uncertainty_status: HistogramUncertaintyStatus::Available,
            corrections: Box::new(HistogramCorrections::new(bins)),
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
        let bins = bin_edges.len().saturating_sub(1);
        let axis = Self::axis_for_edges(&bin_edges)?;
        let histogram = Self {
            fill_kind: HistogramFillKind::Empirical,
            counts: vec![0.0; bins],
            axis,
            underflow: 0.0,
            overflow: 0.0,
            errors: vec![0.0; bins],
            sum_squared_weights: vec![0.0; bins],
            underflow_sum_squared_weights: Some(0.0),
            overflow_sum_squared_weights: Some(0.0),
            uncertainty_status: HistogramUncertaintyStatus::Available,
            corrections: Box::new(HistogramCorrections::new(bins)),
        };
        histogram.validate_requirements(HistogramRequirements::VALID)?;
        Ok(histogram)
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
        self.sum_squared_weights = errors.iter().map(|error| error * error).collect();
        self.corrections.sum_squared_weights.fill(0.0);
        self.fill_kind = HistogramFillKind::Manual;
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
        self.fill_kind = HistogramFillKind::Manual;
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
            Some(FillTarget::Underflow) => {
                Self::add_compensated(&mut self.underflow, &mut self.corrections.underflow, weight);
                if let Some(sum_squared_weights) = &mut self.underflow_sum_squared_weights {
                    Self::add_compensated(
                        sum_squared_weights,
                        &mut self.corrections.underflow_sum_squared_weights,
                        uncertainty * uncertainty,
                    );
                }
            }
            Some(FillTarget::Bin(index)) => {
                Self::add_compensated(
                    &mut self.counts[index],
                    &mut self.corrections.counts[index],
                    weight,
                );
                Self::add_compensated(
                    &mut self.sum_squared_weights[index],
                    &mut self.corrections.sum_squared_weights[index],
                    uncertainty * uncertainty,
                );
                self.errors[index] = self.sum_squared_weights[index].sqrt();
            }
            Some(FillTarget::Overflow) => {
                Self::add_compensated(&mut self.overflow, &mut self.corrections.overflow, weight);
                if let Some(sum_squared_weights) = &mut self.overflow_sum_squared_weights {
                    Self::add_compensated(
                        sum_squared_weights,
                        &mut self.corrections.overflow_sum_squared_weights,
                        uncertainty * uncertainty,
                    );
                }
            }
            None => {}
        }
    }

    fn add_compensated(sum: &mut f64, correction: &mut f64, value: f64) {
        let corrected = value - *correction;
        let next = *sum + corrected;
        *correction = (next - *sum) - corrected;
        *sum = next;
    }

    fn fill_target(&self, value: f64) -> Option<FillTarget> {
        if value < self.axis.edges()[0] {
            Some(FillTarget::Underflow)
        } else if value >= self.axis.edges()[self.axis.edges().len() - 1] {
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
        self.corrections.counts.fill(0.0);
        self.fill_kind = HistogramFillKind::Manual;
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
        self.corrections.counts[bin_index] = 0.0;
        self.fill_kind = HistogramFillKind::Manual;
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
        self.sum_squared_weights[bin_index] = error * error;
        self.corrections.sum_squared_weights[bin_index] = 0.0;
        self.fill_kind = HistogramFillKind::Manual;
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

    /// Return errors only when the stored constituents define an honest uncertainty.
    pub fn reported_errors(&self) -> Option<&[f64]> {
        (self.uncertainty_status == HistogramUncertaintyStatus::Available).then_some(self.errors())
    }

    /// Return whether covariance information is sufficient to report errors.
    pub fn uncertainty_status(&self) -> HistogramUncertaintyStatus {
        self.uncertainty_status
    }

    /// Return the empirical squared-weight constituent for every regular bin.
    pub fn sum_squared_weights(&self) -> &[f64] {
        &self.sum_squared_weights
    }

    /// Return the bin edges.
    pub fn bin_edges(&self) -> &[f64] {
        self.axis.edges()
    }

    /// Return the accumulated underflow weight.
    pub fn underflow(&self) -> f64 {
        self.underflow
    }

    /// Return the empirical squared-weight constituent below the first edge.
    pub fn underflow_sum_squared_weights(&self) -> Option<f64> {
        self.underflow_sum_squared_weights
    }

    /// Return the accumulated overflow weight.
    pub fn overflow(&self) -> f64 {
        self.overflow
    }

    /// Return the empirical squared-weight constituent at or above the final edge.
    pub fn overflow_sum_squared_weights(&self) -> Option<f64> {
        self.overflow_sum_squared_weights
    }

    /// Merge a histogram filled from a disjoint event partition.
    ///
    /// Geometry, empirical/manual fill policy, and flow-constituent availability
    /// must match exactly. A failed merge leaves this histogram unchanged.
    /// The caller is responsible for ensuring the inputs describe disjoint fills;
    /// histogram accumulators do not retain event-source provenance.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when either histogram is invalid, their
    /// geometry or fill policies differ, or merged finite accumulators overflow.
    pub fn merge(&mut self, other: &Self) -> LadduPhysicsResult<()> {
        self.validate_requirements(HistogramRequirements::VALID)?;
        other.validate_requirements(HistogramRequirements::VALID)?;
        if self.axis != other.axis {
            return Err(LadduPhysicsError::invalid_relation(
                "histogram merge requires identical bin edges",
            ));
        }
        if self.fill_kind != other.fill_kind
            || self.underflow_sum_squared_weights.is_some()
                != other.underflow_sum_squared_weights.is_some()
            || self.overflow_sum_squared_weights.is_some()
                != other.overflow_sum_squared_weights.is_some()
        {
            return Err(LadduPhysicsError::invalid_relation(
                "histogram merge requires identical fill and uncertainty policies",
            ));
        }

        let mut merged = self.clone();
        if other.uncertainty_status == HistogramUncertaintyStatus::UnavailableCovariance {
            merged.uncertainty_status = HistogramUncertaintyStatus::UnavailableCovariance;
        }
        for index in 0..merged.counts.len() {
            Self::add_compensated(
                &mut merged.counts[index],
                &mut merged.corrections.counts[index],
                other.counts[index],
            );
            Self::add_compensated(
                &mut merged.counts[index],
                &mut merged.corrections.counts[index],
                -other.corrections.counts[index],
            );
            Self::add_compensated(
                &mut merged.sum_squared_weights[index],
                &mut merged.corrections.sum_squared_weights[index],
                other.sum_squared_weights[index],
            );
            Self::add_compensated(
                &mut merged.sum_squared_weights[index],
                &mut merged.corrections.sum_squared_weights[index],
                -other.corrections.sum_squared_weights[index],
            );
            merged.errors[index] = merged.sum_squared_weights[index].sqrt();
        }
        Self::add_compensated(
            &mut merged.underflow,
            &mut merged.corrections.underflow,
            other.underflow,
        );
        Self::add_compensated(
            &mut merged.underflow,
            &mut merged.corrections.underflow,
            -other.corrections.underflow,
        );
        Self::add_compensated(
            &mut merged.overflow,
            &mut merged.corrections.overflow,
            other.overflow,
        );
        Self::add_compensated(
            &mut merged.overflow,
            &mut merged.corrections.overflow,
            -other.corrections.overflow,
        );
        if let (Some(merged_sum), Some(other_sum)) = (
            &mut merged.underflow_sum_squared_weights,
            other.underflow_sum_squared_weights,
        ) {
            Self::add_compensated(
                merged_sum,
                &mut merged.corrections.underflow_sum_squared_weights,
                other_sum,
            );
            Self::add_compensated(
                merged_sum,
                &mut merged.corrections.underflow_sum_squared_weights,
                -other.corrections.underflow_sum_squared_weights,
            );
        }
        if let (Some(merged_sum), Some(other_sum)) = (
            &mut merged.overflow_sum_squared_weights,
            other.overflow_sum_squared_weights,
        ) {
            Self::add_compensated(
                merged_sum,
                &mut merged.corrections.overflow_sum_squared_weights,
                other_sum,
            );
            Self::add_compensated(
                merged_sum,
                &mut merged.corrections.overflow_sum_squared_weights,
                -other.corrections.overflow_sum_squared_weights,
            );
        }
        merged.validate_requirements(HistogramRequirements::VALID)?;
        *self = merged;
        Ok(())
    }

    /// Add a compatible histogram without assuming the inputs are independent.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, fill policy, or non-finite output.
    pub fn add(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, 1.0, false)
    }

    /// Add a compatible histogram with an explicit caller assertion of independence.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, fill policy, or non-finite output.
    pub fn add_independent(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, 1.0, true)
    }

    /// Subtract a compatible histogram without assuming the inputs are independent.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, fill policy, or non-finite output.
    pub fn subtract(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, -1.0, false)
    }

    /// Subtract a compatible histogram with an explicit caller assertion of independence.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry, fill policy, or non-finite output.
    pub fn subtract_independent(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, -1.0, true)
    }

    fn combine(
        &self,
        other: &Self,
        sign: f64,
        assert_independent: bool,
    ) -> LadduPhysicsResult<Self> {
        self.validate_requirements(HistogramRequirements::VALID)?;
        other.validate_requirements(HistogramRequirements::VALID)?;
        if self.axis != other.axis || self.fill_kind != other.fill_kind {
            return Err(LadduPhysicsError::invalid_relation(
                "histogram arithmetic requires identical bin edges and fill policy",
            ));
        }
        if self.underflow_sum_squared_weights.is_some()
            != other.underflow_sum_squared_weights.is_some()
            || self.overflow_sum_squared_weights.is_some()
                != other.overflow_sum_squared_weights.is_some()
        {
            return Err(LadduPhysicsError::invalid_relation(
                "histogram arithmetic requires matching flow uncertainty constituents",
            ));
        }
        let mut result = self.clone();
        for index in 0..result.counts.len() {
            Self::add_compensated(
                &mut result.counts[index],
                &mut result.corrections.counts[index],
                sign * other.counts[index],
            );
            Self::add_compensated(
                &mut result.sum_squared_weights[index],
                &mut result.corrections.sum_squared_weights[index],
                other.sum_squared_weights[index],
            );
            result.errors[index] = result.sum_squared_weights[index].sqrt();
        }
        Self::add_compensated(
            &mut result.underflow,
            &mut result.corrections.underflow,
            sign * other.underflow,
        );
        Self::add_compensated(
            &mut result.overflow,
            &mut result.corrections.overflow,
            sign * other.overflow,
        );
        result.underflow_sum_squared_weights = self
            .underflow_sum_squared_weights
            .zip(other.underflow_sum_squared_weights)
            .map(|(left, right)| left + right);
        result.overflow_sum_squared_weights = self
            .overflow_sum_squared_weights
            .zip(other.overflow_sum_squared_weights)
            .map(|(left, right)| left + right);
        result.uncertainty_status = if self.uncertainty_status
            == HistogramUncertaintyStatus::Available
            && other.uncertainty_status == HistogramUncertaintyStatus::Available
            && assert_independent
        {
            HistogramUncertaintyStatus::Available
        } else {
            HistogramUncertaintyStatus::UnavailableCovariance
        };
        result.validate_requirements(HistogramRequirements::VALID)?;
        Ok(result)
    }

    /// Scale central values and uncertainty constituents by a finite factor.
    ///
    /// # Errors
    /// Returns an error when the factor or scaled output is non-finite.
    pub fn scaled(&self, factor: f64) -> LadduPhysicsResult<Self> {
        if !factor.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "histogram scale",
                "finite",
                factor,
            ));
        }
        let mut result = self.clone();
        let variance_scale = factor * factor;
        for index in 0..result.counts.len() {
            result.counts[index] *= factor;
            result.sum_squared_weights[index] *= variance_scale;
            result.errors[index] *= factor.abs();
        }
        result.underflow *= factor;
        result.overflow *= factor;
        result.underflow_sum_squared_weights = result
            .underflow_sum_squared_weights
            .map(|v| v * variance_scale);
        result.overflow_sum_squared_weights = result
            .overflow_sum_squared_weights
            .map(|v| v * variance_scale);
        result.corrections = Box::new(HistogramCorrections::new(result.bins()));
        result.validate_requirements(HistogramRequirements::VALID)?;
        Ok(result)
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
        (
            self.axis.edges()[0],
            self.axis.edges()[self.axis.edges().len() - 1],
        )
    }

    /// Return the number of bins.
    pub fn bins(&self) -> usize {
        self.counts.len()
    }

    /// Return the bin index for a value.
    ///
    /// The lower edge is inclusive and the upper edge is exclusive.
    pub fn bin_index(&self, value: f64) -> Option<usize> {
        self.axis.index(value, FinalUpperEdge::Exclusive)
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
                total_weight * (self.axis.edges()[index + 1] - self.axis.edges()[index])
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
        let sum_squared_weights = self
            .sum_squared_weights
            .iter()
            .enumerate()
            .map(|(index, sum_squared_weight)| sum_squared_weight / scale(index).powi(2))
            .collect::<Vec<_>>();
        let errors = sum_squared_weights
            .iter()
            .map(|sum_squared_weight| sum_squared_weight.sqrt())
            .collect();
        let (underflow, overflow) = if policy.preserve_flow {
            (self.underflow / total_weight, self.overflow / total_weight)
        } else {
            (0.0, 0.0)
        };
        let flow_scale = total_weight * total_weight;
        let (underflow_sum_squared_weights, overflow_sum_squared_weights) = if policy.preserve_flow
        {
            (
                self.underflow_sum_squared_weights
                    .map(|sum| sum / flow_scale),
                self.overflow_sum_squared_weights
                    .map(|sum| sum / flow_scale),
            )
        } else if self.fill_kind == HistogramFillKind::Empirical {
            (Some(0.0), Some(0.0))
        } else {
            (None, None)
        };
        let bins = self.counts.len();
        let histogram = Self {
            fill_kind: self.fill_kind,
            counts,
            axis: self.axis.clone(),
            underflow,
            overflow,
            errors,
            sum_squared_weights,
            underflow_sum_squared_weights,
            overflow_sum_squared_weights,
            uncertainty_status: self.uncertainty_status,
            corrections: Box::new(HistogramCorrections::new(bins)),
        };
        histogram.validate_requirements(HistogramRequirements::VALID)?;
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
                let low = self.axis.edges()[i];
                let high = self.axis.edges()[i + 1];
                return Ok(rng.f64_range(low..high));
            }
        }

        // Handles tiny floating-point roundoff.
        let last = self.counts.len() - 1;
        Ok(rng.f64_range(self.axis.edges()[last]..self.axis.edges()[last + 1]))
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
        0.5 * (self.axis.edges()[index] + self.axis.edges()[index + 1])
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
        if self.axis.edges().len() < 2 {
            return Err(LadduPhysicsError::invalid_length(
                "histogram bin edges",
                "at least 2",
                self.axis.edges().len(),
            ));
        }

        if self.counts.len() + 1 != self.axis.edges().len() {
            return Err(LadduPhysicsError::invalid_length(
                "histogram counts/bin_edges",
                "counts.len() + 1 == bin_edges.len()",
                format!(
                    "{} counts and {} edges",
                    self.counts.len(),
                    self.axis.edges().len()
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

        if self.sum_squared_weights.len() != self.counts.len() {
            return Err(LadduPhysicsError::invalid_length(
                "histogram sum_squared_weights",
                format!("same length as counts ({})", self.counts.len()),
                self.sum_squared_weights.len(),
            ));
        }
        if self.corrections.counts.len() != self.counts.len()
            || self.corrections.sum_squared_weights.len() != self.counts.len()
        {
            return Err(LadduPhysicsError::invalid_length(
                "histogram compensated accumulators",
                format!("same length as counts ({})", self.counts.len()),
                format!(
                    "{} count and {} squared-weight corrections",
                    self.corrections.counts.len(),
                    self.corrections.sum_squared_weights.len()
                ),
            ));
        }

        for (index, edges) in self.axis.edges().windows(2).enumerate() {
            if edges[1] <= edges[0] {
                return Err(LadduPhysicsError::invalid_relation(format!(
                    "histogram bin edges must be strictly increasing at edge pair {index}"
                )));
            }
        }

        Ok(())
    }

    fn validate_finite(&self) -> LadduPhysicsResult<()> {
        for (index, edge) in self.axis.edges().iter().enumerate() {
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
        for (index, sum_squared_weights) in self.sum_squared_weights.iter().enumerate() {
            if !sum_squared_weights.is_finite() || *sum_squared_weights < 0.0 {
                return Err(LadduPhysicsError::invalid_value(
                    format!("histogram sum of squared weights {index}"),
                    "finite and nonnegative",
                    *sum_squared_weights,
                ));
            }
        }

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

        if let Some(sum_squared_weights) = self.underflow_sum_squared_weights {
            Self::validate_error(
                "histogram underflow sum of squared weights",
                sum_squared_weights,
            )?;
        }
        if let Some(sum_squared_weights) = self.overflow_sum_squared_weights {
            Self::validate_error(
                "histogram overflow sum of squared weights",
                sum_squared_weights,
            )?;
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
    fn arithmetic_distinguishes_proven_independence_from_shared_sources() {
        let left =
            Histogram::from_values_with_edges(&[0.25], vec![0.0, 1.0], Some(&[2.0])).unwrap();
        let independent =
            Histogram::from_values_with_edges(&[0.25], vec![0.0, 1.0], Some(&[3.0])).unwrap();

        let conservative_sum = left.add(&independent).unwrap();
        assert_eq!(conservative_sum.counts(), [5.0]);
        assert_eq!(
            conservative_sum.uncertainty_status(),
            HistogramUncertaintyStatus::UnavailableCovariance
        );
        assert!(conservative_sum.reported_errors().is_none());

        let correlated_sum = left.add(&left).unwrap();
        assert_eq!(correlated_sum.counts(), [4.0]);
        assert_eq!(
            correlated_sum.uncertainty_status(),
            HistogramUncertaintyStatus::UnavailableCovariance
        );
        assert!(correlated_sum.reported_errors().is_none());

        let asserted = left.add_independent(&independent).unwrap();
        assert_eq!(asserted.errors(), [13.0_f64.sqrt()]);
        let asserted = left.add_independent(&left).unwrap();
        assert_eq!(asserted.reported_errors().unwrap(), [8.0_f64.sqrt()]);
    }

    #[test]
    fn subtraction_scaling_and_serialization_preserve_uncertainty_semantics() {
        let left =
            Histogram::from_values_with_edges(&[0.25], vec![0.0, 1.0], Some(&[-2.0])).unwrap();
        let right =
            Histogram::from_values_with_edges(&[0.25], vec![0.0, 1.0], Some(&[3.0])).unwrap();

        let difference = left.subtract_independent(&right).unwrap();
        assert_eq!(difference.counts(), [-5.0]);
        assert_eq!(difference.reported_errors().unwrap(), [13.0_f64.sqrt()]);
        let scaled = difference.scaled(-2.0).unwrap();
        assert_eq!(scaled.counts(), [10.0]);
        assert_eq!(scaled.sum_squared_weights(), [52.0]);

        let unavailable = left.subtract(&left).unwrap();
        let json = serde_json::to_string(&unavailable).unwrap();
        let restored: Histogram = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.uncertainty_status(),
            HistogramUncertaintyStatus::UnavailableCovariance
        );
        assert!(restored.reported_errors().is_none());
    }

    #[test]
    fn arithmetic_rejects_incompatible_geometry_and_fill_policy() {
        let empirical = Histogram::empty_with_edges(vec![0.0, 1.0]).unwrap();
        let different_edges = Histogram::empty_with_edges(vec![0.0, 2.0]).unwrap();
        let manual = Histogram::new(vec![0.0], vec![0.0, 1.0]).unwrap();

        assert!(empirical.add(&different_edges).is_err());
        assert!(empirical.subtract(&manual).is_err());
        assert!(empirical.scaled(f64::INFINITY).is_err());
    }

    #[test]
    fn repeated_arithmetic_uses_deterministic_compensated_accumulation() {
        let large = Histogram::new(vec![1.0e16], vec![0.0, 1.0]).unwrap();
        let unit = Histogram::new(vec![1.0], vec![0.0, 1.0]).unwrap();

        let result = large
            .add_independent(&unit)
            .unwrap()
            .add_independent(&unit)
            .unwrap();

        assert_eq!(result.counts(), [1.0e16 + 2.0]);
    }

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
    fn disjoint_weighted_histograms_merge_their_empirical_constituents() {
        let mut left = Histogram::empty(2, (0.0, 1.0)).unwrap();
        left.fill_weighted(-0.1, -2.0).unwrap();
        left.fill_weighted(0.25, 3.0).unwrap();

        let mut right = Histogram::empty(2, (0.0, 1.0)).unwrap();
        right.fill_weighted(0.25, -4.0).unwrap();
        right.fill_weighted(0.75, 5.0).unwrap();
        right.fill_weighted(1.0, 6.0).unwrap();

        left.merge(&right).unwrap();

        assert_relative_eq!(left.counts(), &[-1.0, 5.0][..]);
        assert_relative_eq!(left.sum_squared_weights(), &[25.0, 25.0][..]);
        assert_relative_eq!(left.errors(), &[5.0, 5.0][..]);
        assert_relative_eq!(left.underflow(), -2.0);
        assert_relative_eq!(left.underflow_sum_squared_weights().unwrap(), 4.0);
        assert_relative_eq!(left.overflow(), 6.0);
        assert_relative_eq!(left.overflow_sum_squared_weights().unwrap(), 36.0);
    }

    #[test]
    fn incompatible_merge_is_atomic() {
        let mut histogram = Histogram::from_values(&[0.25], 2, (0.0, 1.0), Some(&[2.0])).unwrap();
        let original = histogram.clone();
        let incompatible = Histogram::empty_with_edges(vec![0.0, 0.25, 1.0]).unwrap();

        assert!(histogram.merge(&incompatible).is_err());
        assert_eq!(histogram, original);
    }

    #[test]
    fn merge_rejects_incompatible_fill_policies() {
        let mut empirical = Histogram::empty(2, (0.0, 1.0)).unwrap();
        let manual = Histogram::new(vec![0.0, 0.0], vec![0.0, 0.5, 1.0]).unwrap();
        let original = empirical.clone();

        assert!(empirical.merge(&manual).is_err());
        assert_eq!(empirical, original);
    }

    #[test]
    fn deterministic_disjoint_merges_match_single_pass_and_ignore_empty_chunks() {
        let values = [-0.2, 0.1, 0.4, 0.6, 0.9, 1.2];
        let weights = [2.0, 1.0e16, -1.0e16, 3.0, -4.0, 5.0];
        let single = Histogram::from_values(&values, 2, (0.0, 1.0), Some(&weights)).unwrap();
        let mut merged = Histogram::empty(2, (0.0, 1.0)).unwrap();

        for (value_chunk, weight_chunk) in values.chunks(2).zip(weights.chunks(2)) {
            let partial =
                Histogram::from_values(value_chunk, 2, (0.0, 1.0), Some(weight_chunk)).unwrap();
            merged.merge(&partial).unwrap();
        }
        merged
            .merge(&Histogram::empty(2, (0.0, 1.0)).unwrap())
            .unwrap();

        assert_relative_eq!(merged.counts(), single.counts(), max_relative = 1e-15);
        assert_relative_eq!(
            merged.sum_squared_weights(),
            single.sum_squared_weights(),
            max_relative = 1e-15
        );
        assert_relative_eq!(merged.underflow(), single.underflow(), max_relative = 1e-15);
        assert_relative_eq!(merged.overflow(), single.overflow(), max_relative = 1e-15);
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
    fn normalized_with_flow_scales_empirical_flow_constituents() {
        let histogram =
            Histogram::from_values(&[-0.5, 0.5, 1.5], 1, (0.0, 1.0), Some(&[2.0, 3.0, 4.0]))
                .unwrap();

        let normalized = histogram.normalized_with_flow().unwrap();

        assert_relative_eq!(
            normalized.underflow_sum_squared_weights().unwrap(),
            4.0 / 81.0
        );
        assert_relative_eq!(
            normalized.overflow_sum_squared_weights().unwrap(),
            16.0 / 81.0
        );
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
