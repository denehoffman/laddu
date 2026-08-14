use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::{
    error::{MemoryError, MemoryResult},
    resource::MemoryResource,
};

const AUTO_AVAILABLE_FRACTION: f64 = 0.80;

/// A requested memory limit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBudget {
    /// Automatically use 80% of currently available memory.
    #[default]
    Auto,
    /// An absolute number of bytes.
    Bytes(u64),
    /// A fraction in `(0, 1]` of total physical capacity.
    PercentTotal(f64),
    /// A fraction in `(0, 1]` of currently available capacity.
    PercentAvailable(f64),
}

impl MemoryBudget {
    /// Creates an absolute byte budget.
    pub const fn bytes(bytes: u64) -> Self {
        Self::Bytes(bytes)
    }

    /// Creates a percentage-of-total budget.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::InvalidBudget`] unless `percent` is in `(0, 100]`.
    pub fn percent_total(percent: f64) -> MemoryResult<Self> {
        Ok(Self::PercentTotal(validate_percent(percent)? / 100.0))
    }

    /// Creates a percentage-of-available budget.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::InvalidBudget`] unless `percent` is in `(0, 100]`.
    pub fn percent_available(percent: f64) -> MemoryResult<Self> {
        Ok(Self::PercentAvailable(validate_percent(percent)? / 100.0))
    }

    /// Resolves this request for a resource snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error for zero budgets, invalid percentages, or unavailable
    /// capacity telemetry.
    pub fn resolve(self, resource: &MemoryResource) -> MemoryResult<u64> {
        let resolved = match self {
            Self::Auto => resource
                .available_bytes
                .map(|bytes| scaled_bytes(bytes, AUTO_AVAILABLE_FRACTION))
                .or(resource.total_bytes.map(|bytes| scaled_bytes(bytes, 0.5)))
                .ok_or_else(|| MemoryError::UnknownCapacity {
                    resource: resource.name.clone(),
                    budget: self,
                    basis: "available",
                })?,
            Self::Bytes(bytes) => bytes,
            Self::PercentTotal(fraction) => {
                validate_fraction(fraction)?;
                scaled_bytes(
                    resource
                        .total_bytes
                        .ok_or_else(|| MemoryError::UnknownCapacity {
                            resource: resource.name.clone(),
                            budget: self,
                            basis: "total",
                        })?,
                    fraction,
                )
            }
            Self::PercentAvailable(fraction) => {
                validate_fraction(fraction)?;
                scaled_bytes(
                    resource
                        .available_bytes
                        .ok_or_else(|| MemoryError::UnknownCapacity {
                            resource: resource.name.clone(),
                            budget: self,
                            basis: "available",
                        })?,
                    fraction,
                )
            }
        };
        if resolved == 0 {
            return Err(MemoryError::InvalidBudget(
                "resolved budget must be greater than zero".into(),
            ));
        }
        Ok(resolved)
    }
}

impl fmt::Display for MemoryBudget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => formatter.write_str("auto"),
            Self::Bytes(bytes) => write!(formatter, "{bytes} B"),
            Self::PercentTotal(value) => write!(formatter, "{}% total", value * 100.0),
            Self::PercentAvailable(value) => write!(formatter, "{}% available", value * 100.0),
        }
    }
}

impl FromStr for MemoryBudget {
    type Err = MemoryError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let normalized = input.trim().to_ascii_lowercase();
        if normalized == "auto" {
            return Ok(Self::Auto);
        }
        if let Some((percent, suffix)) = normalized.split_once('%') {
            let percent = percent
                .trim()
                .parse::<f64>()
                .map_err(|_| MemoryError::InvalidBudget(input.into()))?;
            return match suffix.trim() {
                "" | "total" => Self::percent_total(percent),
                "available" | "free" | "remaining" => Self::percent_available(percent),
                _ => Err(MemoryError::InvalidBudget(input.into())),
            };
        }
        parse_bytes(&normalized).map(Self::Bytes)
    }
}

/// Host and optional accelerator budgets for one execution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    /// Host allocations, including source and staging buffers.
    pub host: MemoryBudget,
    /// Device allocations. Required only for accelerator execution.
    pub device: Option<MemoryBudget>,
}

impl Default for MemoryPlan {
    fn default() -> Self {
        Self {
            host: MemoryBudget::Auto,
            device: Some(MemoryBudget::Auto),
        }
    }
}

impl MemoryPlan {
    /// Creates a host-only plan.
    pub const fn host(host: MemoryBudget) -> Self {
        Self { host, device: None }
    }
    /// Creates a host-and-device plan.
    pub const fn host_device(host: MemoryBudget, device: MemoryBudget) -> Self {
        Self {
            host,
            device: Some(device),
        }
    }
}

fn validate_percent(percent: f64) -> MemoryResult<f64> {
    if percent.is_finite() && percent > 0.0 && percent <= 100.0 {
        Ok(percent)
    } else {
        Err(MemoryError::InvalidBudget(
            "percentage must be finite and in (0, 100]".into(),
        ))
    }
}

fn validate_fraction(fraction: f64) -> MemoryResult<()> {
    validate_percent(fraction * 100.0).map(|_| ())
}
fn scaled_bytes(bytes: u64, fraction: f64) -> u64 {
    ((bytes as f64) * fraction).floor().min(u64::MAX as f64) as u64
}

fn parse_bytes(input: &str) -> MemoryResult<u64> {
    let split = input
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(input.len());
    let (number, unit) = input.split_at(split);
    let multiplier = match unit.trim() {
        "" | "b" | "byte" | "bytes" => 1,
        "kb" => 1_000,
        "mb" => 1_000_000,
        "gb" => 1_000_000_000,
        "tb" => 1_000_000_000_000,
        "kib" => 1 << 10,
        "mib" => 1 << 20,
        "gib" => 1 << 30,
        "tib" => 1 << 40,
        _ => return Err(MemoryError::InvalidBudget(input.into())),
    };
    let (whole, fraction) = match number.split_once('.') {
        Some((whole, fraction)) if !fraction.contains('.') => (whole, fraction),
        Some(_) => return Err(MemoryError::InvalidBudget(input.into())),
        None => (number, ""),
    };
    if whole.is_empty() && fraction.is_empty()
        || !whole.bytes().all(|digit| digit.is_ascii_digit())
        || !fraction.bytes().all(|digit| digit.is_ascii_digit())
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .any(|digit| digit != b'0')
    {
        return Err(MemoryError::InvalidBudget(input.into()));
    }
    let whole = if whole.is_empty() {
        0
    } else {
        whole
            .parse::<u64>()
            .map_err(|_| MemoryError::InvalidBudget(input.into()))?
    };
    let whole_bytes = whole
        .checked_mul(multiplier)
        .ok_or_else(|| MemoryError::InvalidBudget(input.into()))?;
    let fractional_bytes = fraction.bytes().rev().try_fold(0_u64, |carry, digit| {
        u64::from(digit - b'0')
            .checked_mul(multiplier)
            .and_then(|value| value.checked_add(carry))
            .map(|value| value / 10)
    });
    whole_bytes
        .checked_add(fractional_bytes.ok_or_else(|| MemoryError::InvalidBudget(input.into()))?)
        .ok_or_else(|| MemoryError::InvalidBudget(input.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CapacitySource, MemoryResourceKind};

    fn resource() -> MemoryResource {
        MemoryResource {
            id: "test".into(),
            name: "Test".into(),
            kind: MemoryResourceKind::Device,
            total_bytes: Some(1_000),
            available_bytes: Some(500),
            capacity_source: CapacitySource::User,
            device_identity: None,
        }
    }

    #[test]
    fn parses_and_resolves_budgets() {
        assert_eq!(
            "8 GiB".parse(),
            Ok(MemoryBudget::Bytes(8 * 1024_u64.pow(3)))
        );
        assert_eq!("70% total".parse(), Ok(MemoryBudget::PercentTotal(0.7)));
        assert_eq!(
            "60% available".parse(),
            Ok(MemoryBudget::PercentAvailable(0.6))
        );
        assert_eq!(MemoryBudget::Auto.resolve(&resource()), Ok(400));
        assert_eq!(
            MemoryBudget::PercentTotal(0.5).resolve(&resource()),
            Ok(500)
        );
        assert_eq!(
            MemoryBudget::PercentAvailable(0.5).resolve(&resource()),
            Ok(250)
        );
    }

    #[test]
    fn parses_absolute_budgets_exactly_at_boundaries() {
        for (unit, expected) in [
            ("b", 1),
            ("byte", 1),
            ("bytes", 1),
            ("kb", 1_000),
            ("mb", 1_000_000),
            ("gb", 1_000_000_000),
            ("tb", 1_000_000_000_000),
            ("kib", 1 << 10),
            ("mib", 1 << 20),
            ("gib", 1 << 30),
            ("tib", 1 << 40),
        ] {
            assert_eq!(
                format!("1 {unit}").parse(),
                Ok(MemoryBudget::Bytes(expected))
            );
        }
        assert_eq!("1.5 KiB".parse(), Ok(MemoryBudget::Bytes(1_536)));
        assert_eq!(".5 kb".parse(), Ok(MemoryBudget::Bytes(500)));
        assert_eq!("1.999 B".parse(), Ok(MemoryBudget::Bytes(1)));
        assert_eq!("0.1 B".parse(), Ok(MemoryBudget::Bytes(0)));
        assert_eq!(
            "9007199254740991 B".parse(),
            Ok(MemoryBudget::Bytes(9_007_199_254_740_991))
        );
        assert_eq!(
            "9007199254740993 B".parse(),
            Ok(MemoryBudget::Bytes(9_007_199_254_740_993))
        );
        assert_eq!(
            "18446744073709551615 B".parse(),
            Ok(MemoryBudget::Bytes(u64::MAX))
        );
        assert_eq!(
            "18446744073709551.615 KB".parse(),
            Ok(MemoryBudget::Bytes(u64::MAX))
        );
        assert_eq!(
            MemoryBudget::Bytes(u64::MAX).to_string().parse(),
            Ok(MemoryBudget::Bytes(u64::MAX))
        );
    }

    #[test]
    fn rejects_invalid_budgets() {
        for input in [
            "",
            "0",
            ".",
            "NaN",
            "inf",
            "1.2.3 B",
            "1 XB",
            "bytes",
            "18446744073709551616 B",
            "18446744073709551.616 KB",
        ] {
            assert!(
                input.parse::<MemoryBudget>().is_err(),
                "{input:?} should be invalid"
            );
        }
        for percent in [0.0, -1.0, 100.1, f64::NAN, f64::INFINITY] {
            assert!(MemoryBudget::percent_total(percent).is_err());
            assert!(MemoryBudget::percent_available(percent).is_err());
        }
        assert_eq!(
            MemoryBudget::percent_total(100.0),
            Ok(MemoryBudget::PercentTotal(1.0))
        );
    }
}
