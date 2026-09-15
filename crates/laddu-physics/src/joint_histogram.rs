use serde::{Deserialize, Serialize};

use crate::{
    LadduPhysicsError, LadduPhysicsResult,
    binning::{BinningAxis, FinalUpperEdge, bin_shape, flat_bin_index},
    histogram::{HistogramUncertaintyStatus, uncertainty_is_available},
};

/// Aggregate diagnostics for events that could not be placed in a joint histogram.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct JointHistogramDiagnostics {
    nonfinite_count: u64,
    nonfinite_weight: f64,
    out_of_range_count: u64,
    out_of_range_weight: f64,
}

impl JointHistogramDiagnostics {
    /// Number of events with a nonfinite coordinate or weight.
    pub fn nonfinite_count(&self) -> u64 {
        self.nonfinite_count
    }
    /// Sum of finite event weights classified as nonfinite.
    pub fn nonfinite_weight(&self) -> f64 {
        self.nonfinite_weight
    }
    /// Number of finite events outside at least one axis.
    pub fn out_of_range_count(&self) -> u64 {
        self.out_of_range_count
    }
    /// Sum of weights for finite events outside at least one axis.
    pub fn out_of_range_weight(&self) -> f64 {
        self.out_of_range_weight
    }
}

/// An N-dimensional empirical weighted histogram stored in row-major order.
///
/// The last axis varies fastest. Bins are half-open on both ordinary and final
/// upper edges, matching [`crate::histogram::Histogram`] flow semantics.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointHistogram {
    axes: Vec<Vec<f64>>,
    #[serde(skip)]
    bin_axes: Vec<BinningAxis>,
    shape: Vec<usize>,
    values: Vec<f64>,
    sum_squared_weights: Vec<f64>,
    diagnostics: JointHistogramDiagnostics,
    #[serde(default, skip_serializing_if = "uncertainty_is_available")]
    uncertainty_status: HistogramUncertaintyStatus,
    #[serde(skip)]
    value_corrections: Vec<f64>,
    #[serde(skip)]
    squared_weight_corrections: Vec<f64>,
}

#[derive(Deserialize)]
struct SerializedJointHistogram {
    axes: Vec<Vec<f64>>,
    shape: Vec<usize>,
    values: Vec<f64>,
    sum_squared_weights: Vec<f64>,
    diagnostics: JointHistogramDiagnostics,
    #[serde(default)]
    uncertainty_status: HistogramUncertaintyStatus,
}

impl<'de> Deserialize<'de> for JointHistogram {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let serialized = SerializedJointHistogram::deserialize(deserializer)?;
        let mut histogram = Self::empty(serialized.axes).map_err(serde::de::Error::custom)?;
        if histogram.shape != serialized.shape
            || histogram.values.len() != serialized.values.len()
            || histogram.sum_squared_weights.len() != serialized.sum_squared_weights.len()
        {
            return Err(serde::de::Error::custom(
                "joint histogram shape and flattened accumulator lengths are inconsistent",
            ));
        }
        if serialized
            .values
            .iter()
            .chain(&serialized.sum_squared_weights)
            .any(|value| !value.is_finite())
            || serialized
                .sum_squared_weights
                .iter()
                .any(|value| *value < 0.0)
            || !serialized.diagnostics.nonfinite_weight.is_finite()
            || !serialized.diagnostics.out_of_range_weight.is_finite()
        {
            return Err(serde::de::Error::custom(
                "joint histogram accumulators and diagnostic weights must be finite",
            ));
        }
        histogram.values = serialized.values;
        histogram.sum_squared_weights = serialized.sum_squared_weights;
        histogram.diagnostics = serialized.diagnostics;
        histogram.uncertainty_status = serialized.uncertainty_status;
        Ok(histogram)
    }
}

impl JointHistogram {
    /// Construct an empty joint histogram from ordered axis edges.
    ///
    /// # Errors
    ///
    /// Returns an error for no axes, malformed axis edges, shape overflow, or
    /// an accumulator allocation that cannot be satisfied.
    pub fn empty(axes: Vec<Vec<f64>>) -> LadduPhysicsResult<Self> {
        if axes.is_empty() {
            return Err(LadduPhysicsError::invalid_length(
                "joint histogram axes",
                "at least 1",
                0,
            ));
        }
        let bin_axes = axes
            .iter()
            .enumerate()
            .map(|(axis, edges)| {
                BinningAxis::new(edges.iter().copied()).map_err(|_| {
                    LadduPhysicsError::invalid_relation(format!(
                        "joint histogram axis {axis} edges must contain at least two finite, strictly increasing values"
                    ))
                })
            })
            .collect::<LadduPhysicsResult<Vec<_>>>()?;
        let shape = bin_shape(&bin_axes);
        let bins = checked_bin_count(&shape)?;
        let values = zeroed(bins)?;
        let sum_squared_weights = zeroed(bins)?;
        let value_corrections = zeroed(bins)?;
        let squared_weight_corrections = zeroed(bins)?;
        Ok(Self {
            axes,
            bin_axes,
            shape,
            values,
            sum_squared_weights,
            diagnostics: JointHistogramDiagnostics::default(),
            uncertainty_status: HistogramUncertaintyStatus::Available,
            value_corrections,
            squared_weight_corrections,
        })
    }

    /// Ordered axis-edge metadata.
    pub fn axes(&self) -> &[Vec<f64>] {
        &self.axes
    }
    /// Bin count along each ordered axis.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    /// Row-major flattened weighted sums.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    /// Row-major flattened squared-weight sums.
    pub fn sum_squared_weights(&self) -> &[f64] {
        &self.sum_squared_weights
    }
    /// Row-major flattened empirical standard errors.
    pub fn errors(&self) -> Vec<f64> {
        self.sum_squared_weights
            .iter()
            .map(|value| value.sqrt())
            .collect()
    }
    /// Return errors only when the stored constituents define an honest uncertainty.
    pub fn reported_errors(&self) -> Option<Vec<f64>> {
        (self.uncertainty_status == HistogramUncertaintyStatus::Available).then(|| self.errors())
    }
    /// Return whether covariance information is sufficient to report errors.
    pub fn uncertainty_status(&self) -> HistogramUncertaintyStatus {
        self.uncertainty_status
    }
    /// Aggregate invalid-event diagnostics.
    pub fn diagnostics(&self) -> &JointHistogramDiagnostics {
        &self.diagnostics
    }

    /// Fill one ordered coordinate using an empirical event weight.
    ///
    /// # Errors
    ///
    /// Returns an error when the coordinate rank differs from the histogram
    /// rank or a finite weight's square cannot be represented.
    pub fn fill_weighted(&mut self, coordinates: &[f64], weight: f64) -> LadduPhysicsResult<()> {
        if coordinates.len() != self.axes.len() {
            return Err(LadduPhysicsError::invalid_length(
                "joint histogram coordinates",
                self.axes.len().to_string(),
                coordinates.len(),
            ));
        }
        if coordinates.iter().any(|value| !value.is_finite()) || !weight.is_finite() {
            self.diagnostics.nonfinite_count += 1;
            if weight.is_finite() {
                let next_weight = self.diagnostics.nonfinite_weight + weight;
                if !next_weight.is_finite() {
                    return Err(LadduPhysicsError::invalid_relation(
                        "joint histogram nonfinite diagnostic weight overflow",
                    ));
                }
                self.diagnostics.nonfinite_weight = next_weight;
            }
            return Ok(());
        }
        let squared_weight = weight * weight;
        if !squared_weight.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "joint histogram squared weight",
                "finite",
                squared_weight,
            ));
        }
        let Some(flat) = flat_bin_index(&self.bin_axes, coordinates, FinalUpperEdge::Exclusive)
        else {
            self.diagnostics.out_of_range_count += 1;
            let next_weight = self.diagnostics.out_of_range_weight + weight;
            if !next_weight.is_finite() {
                return Err(LadduPhysicsError::invalid_relation(
                    "joint histogram out-of-range weight overflow",
                ));
            }
            self.diagnostics.out_of_range_weight = next_weight;
            return Ok(());
        };
        if self.value_corrections.len() != self.values.len() {
            self.value_corrections.resize(self.values.len(), 0.0);
        }
        if self.squared_weight_corrections.len() != self.values.len() {
            self.squared_weight_corrections
                .resize(self.values.len(), 0.0);
        }
        compensated_add(
            &mut self.values[flat],
            &mut self.value_corrections[flat],
            weight,
        );
        compensated_add(
            &mut self.sum_squared_weights[flat],
            &mut self.squared_weight_corrections[flat],
            squared_weight,
        );
        if !self.values[flat].is_finite() || !self.sum_squared_weights[flat].is_finite() {
            return Err(LadduPhysicsError::invalid_relation(
                "joint histogram fill produced a non-finite accumulator",
            ));
        }
        Ok(())
    }

    /// Merge a histogram filled from a disjoint event partition atomically.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible geometry, count overflow, or a
    /// nonfinite merged accumulator. A failure leaves `self` unchanged.
    pub fn merge(&mut self, other: &Self) -> LadduPhysicsResult<()> {
        if self.axes != other.axes || self.shape != other.shape {
            return Err(LadduPhysicsError::invalid_relation(
                "joint histogram merge requires identical ordered axes and shape",
            ));
        }
        let mut merged = self.clone();
        if other.uncertainty_status == HistogramUncertaintyStatus::UnavailableCovariance {
            merged.uncertainty_status = HistogramUncertaintyStatus::UnavailableCovariance;
        }
        merged.value_corrections.resize(merged.values.len(), 0.0);
        merged
            .squared_weight_corrections
            .resize(merged.values.len(), 0.0);
        for index in 0..merged.values.len() {
            compensated_add(
                &mut merged.values[index],
                &mut merged.value_corrections[index],
                other.values[index],
            );
            compensated_add(
                &mut merged.sum_squared_weights[index],
                &mut merged.squared_weight_corrections[index],
                other.sum_squared_weights[index],
            );
        }
        merged.diagnostics.nonfinite_count = merged
            .diagnostics
            .nonfinite_count
            .checked_add(other.diagnostics.nonfinite_count)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation("joint histogram diagnostic count overflow")
            })?;
        merged.diagnostics.out_of_range_count = merged
            .diagnostics
            .out_of_range_count
            .checked_add(other.diagnostics.out_of_range_count)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation("joint histogram diagnostic count overflow")
            })?;
        merged.diagnostics.nonfinite_weight += other.diagnostics.nonfinite_weight;
        merged.diagnostics.out_of_range_weight += other.diagnostics.out_of_range_weight;
        if merged
            .values
            .iter()
            .chain(&merged.sum_squared_weights)
            .any(|value| !value.is_finite())
            || !merged.diagnostics.nonfinite_weight.is_finite()
            || !merged.diagnostics.out_of_range_weight.is_finite()
        {
            return Err(LadduPhysicsError::invalid_relation(
                "joint histogram merge produced a non-finite accumulator",
            ));
        }
        *self = merged;
        Ok(())
    }

    /// Add a compatible histogram without assuming the inputs are independent.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry or non-finite output.
    pub fn add(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, 1.0, false)
    }
    /// Add a compatible histogram with an explicit caller assertion of independence.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry or non-finite output.
    pub fn add_independent(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, 1.0, true)
    }
    /// Subtract a compatible histogram without assuming the inputs are independent.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry or non-finite output.
    pub fn subtract(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, -1.0, false)
    }
    /// Subtract a compatible histogram with an explicit caller assertion of independence.
    ///
    /// # Errors
    /// Returns an error for incompatible geometry or non-finite output.
    pub fn subtract_independent(&self, other: &Self) -> LadduPhysicsResult<Self> {
        self.combine(other, -1.0, true)
    }

    fn combine(
        &self,
        other: &Self,
        sign: f64,
        assert_independent: bool,
    ) -> LadduPhysicsResult<Self> {
        if self.axes != other.axes || self.shape != other.shape {
            return Err(LadduPhysicsError::invalid_relation(
                "joint histogram arithmetic requires identical ordered axes and shape",
            ));
        }
        let mut result = self.clone();
        result.value_corrections.resize(result.values.len(), 0.0);
        result
            .squared_weight_corrections
            .resize(result.values.len(), 0.0);
        for index in 0..result.values.len() {
            compensated_add(
                &mut result.values[index],
                &mut result.value_corrections[index],
                sign * other.values[index],
            );
            compensated_add(
                &mut result.sum_squared_weights[index],
                &mut result.squared_weight_corrections[index],
                other.sum_squared_weights[index],
            );
        }
        result.diagnostics.nonfinite_count = result
            .diagnostics
            .nonfinite_count
            .checked_add(other.diagnostics.nonfinite_count)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation("joint histogram diagnostic count overflow")
            })?;
        result.diagnostics.out_of_range_count = result
            .diagnostics
            .out_of_range_count
            .checked_add(other.diagnostics.out_of_range_count)
            .ok_or_else(|| {
                LadduPhysicsError::invalid_relation("joint histogram diagnostic count overflow")
            })?;
        result.diagnostics.nonfinite_weight += sign * other.diagnostics.nonfinite_weight;
        result.diagnostics.out_of_range_weight += sign * other.diagnostics.out_of_range_weight;
        result.uncertainty_status = if self.uncertainty_status
            == HistogramUncertaintyStatus::Available
            && other.uncertainty_status == HistogramUncertaintyStatus::Available
            && assert_independent
        {
            HistogramUncertaintyStatus::Available
        } else {
            HistogramUncertaintyStatus::UnavailableCovariance
        };
        if result
            .values
            .iter()
            .chain(&result.sum_squared_weights)
            .any(|value| !value.is_finite())
            || !result.diagnostics.nonfinite_weight.is_finite()
            || !result.diagnostics.out_of_range_weight.is_finite()
        {
            return Err(LadduPhysicsError::invalid_relation(
                "joint histogram arithmetic produced a non-finite accumulator",
            ));
        }
        Ok(result)
    }

    /// Scale central values, diagnostic weights, and uncertainty constituents.
    ///
    /// # Errors
    /// Returns an error when the factor or scaled output is non-finite.
    pub fn scaled(&self, factor: f64) -> LadduPhysicsResult<Self> {
        if !factor.is_finite() {
            return Err(LadduPhysicsError::invalid_value(
                "joint histogram scale",
                "finite",
                factor,
            ));
        }
        let mut result = self.clone();
        let variance_scale = factor * factor;
        for value in &mut result.values {
            *value *= factor;
        }
        for value in &mut result.sum_squared_weights {
            *value *= variance_scale;
        }
        result.diagnostics.nonfinite_weight *= factor;
        result.diagnostics.out_of_range_weight *= factor;
        result.value_corrections.fill(0.0);
        result.squared_weight_corrections.fill(0.0);
        if result
            .values
            .iter()
            .chain(&result.sum_squared_weights)
            .any(|value| !value.is_finite())
            || !result.diagnostics.nonfinite_weight.is_finite()
            || !result.diagnostics.out_of_range_weight.is_finite()
        {
            return Err(LadduPhysicsError::invalid_relation(
                "joint histogram scaling produced a non-finite accumulator",
            ));
        }
        Ok(result)
    }
}

fn zeroed(len: usize) -> LadduPhysicsResult<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| {
        LadduPhysicsError::invalid_relation(format!(
            "joint histogram shape with {len} bins cannot be allocated"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn checked_bin_count(shape: &[usize]) -> LadduPhysicsResult<usize> {
    shape.iter().try_fold(1usize, |bins, axis_bins| {
        bins.checked_mul(*axis_bins).ok_or_else(|| {
            LadduPhysicsError::invalid_relation("joint histogram shape overflows usize")
        })
    })
}

fn compensated_add(sum: &mut f64, correction: &mut f64, value: f64) {
    let adjusted = value - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_distinguishes_proven_independence_from_shared_sources() {
        let mut left = JointHistogram::empty(vec![vec![0.0, 1.0]]).unwrap();
        left.fill_weighted(&[0.5], 2.0).unwrap();
        let mut independent = JointHistogram::empty(vec![vec![0.0, 1.0]]).unwrap();
        independent.fill_weighted(&[0.5], 3.0).unwrap();

        let conservative_sum = left.add(&independent).unwrap();
        assert_eq!(conservative_sum.values(), [5.0]);
        assert_eq!(
            conservative_sum.uncertainty_status(),
            HistogramUncertaintyStatus::UnavailableCovariance
        );
        assert!(conservative_sum.reported_errors().is_none());

        let asserted = left.add_independent(&independent).unwrap();
        assert_eq!(asserted.reported_errors().unwrap(), [13.0_f64.sqrt()]);

        let correlated_sum = left.add(&left).unwrap();
        assert_eq!(correlated_sum.values(), [4.0]);
        assert_eq!(
            correlated_sum.uncertainty_status(),
            HistogramUncertaintyStatus::UnavailableCovariance
        );
        assert!(correlated_sum.reported_errors().is_none());
    }

    #[test]
    fn arithmetic_scales_serializes_and_rejects_incompatible_axes() {
        let mut histogram = JointHistogram::empty(vec![vec![0.0, 1.0]]).unwrap();
        histogram.fill_weighted(&[0.5], -2.0).unwrap();
        let scaled = histogram.scaled(-3.0).unwrap();
        assert_eq!(scaled.values(), [6.0]);
        assert_eq!(scaled.sum_squared_weights(), [36.0]);

        let unavailable = histogram.subtract(&histogram).unwrap();
        let restored: JointHistogram =
            serde_json::from_str(&serde_json::to_string(&unavailable).unwrap()).unwrap();
        assert_eq!(
            restored.uncertainty_status(),
            HistogramUncertaintyStatus::UnavailableCovariance
        );
        assert!(restored.reported_errors().is_none());

        let incompatible = JointHistogram::empty(vec![vec![0.0, 2.0]]).unwrap();
        assert!(histogram.add(&incompatible).is_err());
        assert!(histogram.scaled(f64::NAN).is_err());
    }

    #[test]
    fn repeated_arithmetic_uses_deterministic_compensated_accumulation() {
        let mut large = JointHistogram::empty(vec![vec![0.0, 1.0]]).unwrap();
        large.fill_weighted(&[0.5], 1.0e16).unwrap();
        let mut unit = JointHistogram::empty(vec![vec![0.0, 1.0]]).unwrap();
        unit.fill_weighted(&[0.5], 1.0).unwrap();

        let result = large
            .add_independent(&unit)
            .unwrap()
            .add_independent(&unit)
            .unwrap();

        assert_eq!(result.values(), [1.0e16 + 2.0]);
    }

    #[test]
    fn validates_shape_and_merges_disjoint_fills() {
        let axes = vec![vec![0.0, 1.0, 2.0], vec![0.0, 5.0, 10.0]];
        let mut left = JointHistogram::empty(axes.clone()).unwrap();
        let mut right = JointHistogram::empty(axes).unwrap();
        left.fill_weighted(&[0.5, 7.0], -2.0).unwrap();
        right.fill_weighted(&[1.5, 2.0], 3.0).unwrap();
        left.merge(&right).unwrap();
        assert_eq!(left.values(), [0.0, -2.0, 3.0, 0.0]);
        assert_eq!(left.errors(), [0.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn final_upper_edges_are_out_of_range_and_nonfinite_takes_precedence() {
        let mut histogram = JointHistogram::empty(vec![vec![0.0, 1.0], vec![0.0, 1.0]]).unwrap();
        histogram.fill_weighted(&[1.0, 0.5], -2.0).unwrap();
        histogram.fill_weighted(&[f64::NAN, 2.0], 3.0).unwrap();

        assert_eq!(histogram.values(), [0.0]);
        assert_eq!(histogram.diagnostics().out_of_range_count(), 1);
        assert_eq!(histogram.diagnostics().out_of_range_weight(), -2.0);
        assert_eq!(histogram.diagnostics().nonfinite_count(), 1);
        assert_eq!(histogram.diagnostics().nonfinite_weight(), 3.0);
    }

    #[test]
    fn nonuniform_lower_and_internal_boundaries_follow_half_open_policy() {
        let mut histogram =
            JointHistogram::empty(vec![vec![0.0, 0.25, 2.0], vec![-1.0, 0.0, 0.5, 3.0]]).unwrap();
        histogram.fill_weighted(&[0.0, -1.0], 1.0).unwrap();
        histogram.fill_weighted(&[0.25, 0.0], 2.0).unwrap();

        assert_eq!(histogram.shape(), [2, 3]);
        assert_eq!(histogram.values(), [1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn incompatible_merge_is_atomic() {
        let mut left = JointHistogram::empty(vec![vec![0.0, 1.0]]).unwrap();
        left.fill_weighted(&[0.5], 2.0).unwrap();
        let before = left.clone();
        let other = JointHistogram::empty(vec![vec![0.0, 2.0]]).unwrap();

        assert!(left.merge(&other).is_err());
        assert_eq!(left, before);
    }

    #[test]
    fn all_out_of_range_and_empty_histograms_remain_valid() {
        let mut histogram = JointHistogram::empty(vec![vec![0.0, 1.0], vec![0.0, 1.0]]).unwrap();
        assert_eq!(histogram.values(), [0.0]);
        histogram.fill_weighted(&[-1.0, 0.5], 2.0).unwrap();
        histogram.fill_weighted(&[0.5, 2.0], -3.0).unwrap();
        assert_eq!(histogram.values(), [0.0]);
        assert_eq!(histogram.diagnostics().out_of_range_count(), 2);
        assert_eq!(histogram.diagnostics().out_of_range_weight(), -1.0);
    }

    #[test]
    fn serialization_restores_fill_and_merge_capability() {
        let mut histogram = JointHistogram::empty(vec![vec![0.0, 1.0]]).unwrap();
        histogram.fill_weighted(&[0.5], 2.0).unwrap();
        let json = serde_json::to_string(&histogram).unwrap();
        let mut restored: JointHistogram = serde_json::from_str(&json).unwrap();
        restored.merge(&histogram).unwrap();
        assert_eq!(restored.values(), [4.0]);
        assert_eq!(restored.sum_squared_weights(), [8.0]);
    }

    #[test]
    fn shape_overflow_is_rejected_before_allocation() {
        assert!(checked_bin_count(&[usize::MAX, 2]).is_err());
    }
}
