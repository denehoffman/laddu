use num::complex::Complex64;
use thiserror::Error;

/// The scalar operation applied to each model value before reduction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReductionTransform {
    /// Use the real component directly.
    Real,
    /// Require a positive real component and use it directly.
    PositiveReal,
    /// Require a positive real component and use its natural logarithm.
    LogPositiveReal,
}

/// A backend-neutral description of a weighted dataset reduction.
///
/// Each event contributes `event_weight * transform(model_value)` to the sum.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ReductionPlan {
    transform: ReductionTransform,
}

/// A transformed model value and its derivative with respect to the real model output.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReductionOutput {
    value: f64,
    derivative: f64,
}

impl ReductionOutput {
    /// Returns the transformed value.
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Returns the derivative with respect to the real model output.
    pub const fn derivative(self) -> f64 {
        self.derivative
    }

    /// Returns the transformed value and derivative.
    pub const fn into_parts(self) -> (f64, f64) {
        (self.value, self.derivative)
    }
}

/// Errors produced while applying a dataset reduction transform.
#[derive(Copy, Clone, Debug, Error, PartialEq)]
pub enum ReductionError {
    /// A positivity-requiring transform received a non-positive value.
    #[error("{transform:?} reduction requires a positive real value, got {value}")]
    NonPositiveValue {
        /// Transform that rejected the value.
        transform: ReductionTransform,
        /// Rejected real value.
        value: f64,
    },
}

impl ReductionPlan {
    /// Creates a weighted reduction using `transform`.
    pub const fn weighted(transform: ReductionTransform) -> Self {
        Self { transform }
    }

    /// Creates a weighted reduction of the real component.
    pub const fn weighted_real() -> Self {
        Self::weighted(ReductionTransform::Real)
    }

    /// Creates a weighted reduction requiring a positive real component.
    pub const fn weighted_positive_real() -> Self {
        Self::weighted(ReductionTransform::PositiveReal)
    }

    /// Creates a weighted reduction of the log of a positive real component.
    pub const fn weighted_log_positive_real() -> Self {
        Self::weighted(ReductionTransform::LogPositiveReal)
    }

    /// Returns the per-event transform.
    pub const fn transform(self) -> ReductionTransform {
        self.transform
    }

    /// Transform one model value and return its chain-rule derivative.
    pub fn apply(self, value: Complex64) -> Result<ReductionOutput, ReductionError> {
        let real = value.re;
        let (value, derivative) = match self.transform {
            ReductionTransform::Real => (real, 1.0),
            ReductionTransform::PositiveReal if real > 0.0 => (real, 1.0),
            ReductionTransform::LogPositiveReal if real > 0.0 => (real.ln(), real.recip()),
            transform
            @ (ReductionTransform::PositiveReal | ReductionTransform::LogPositiveReal) => {
                return Err(ReductionError::NonPositiveValue {
                    transform,
                    value: real,
                });
            }
        };
        Ok(ReductionOutput { value, derivative })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_select_the_expected_transform() {
        assert_eq!(
            ReductionPlan::weighted_real().transform(),
            ReductionTransform::Real
        );
        assert_eq!(
            ReductionPlan::weighted_positive_real().transform(),
            ReductionTransform::PositiveReal
        );
        assert_eq!(
            ReductionPlan::weighted_log_positive_real().transform(),
            ReductionTransform::LogPositiveReal
        );
    }

    #[test]
    fn apply_returns_the_transformed_value_and_derivative() {
        assert_eq!(
            ReductionPlan::weighted_real()
                .apply(Complex64::new(3.0, 4.0))
                .unwrap()
                .into_parts(),
            (3.0, 1.0)
        );
        assert_eq!(
            ReductionPlan::weighted_log_positive_real()
                .apply(Complex64::from(2.0))
                .unwrap()
                .into_parts(),
            (2.0_f64.ln(), 0.5)
        );
        assert_eq!(
            ReductionPlan::weighted_positive_real().apply(Complex64::from(-2.0)),
            Err(ReductionError::NonPositiveValue {
                transform: ReductionTransform::PositiveReal,
                value: -2.0,
            })
        );
    }
}
