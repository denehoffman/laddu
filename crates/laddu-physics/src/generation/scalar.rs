#[derive(Clone, Debug)]
/// Source distribution for a generated scalar value.
pub enum ScalarSource {
    /// A deterministic value.
    Constant(f64),
    /// A uniform distribution on `[low, high)`.
    Uniform {
        /// Lower source bound.
        low: f64,
        /// Upper source bound.
        high: f64,
    },
    /// A piecewise-constant histogram distribution.
    Histogram(Histogram),
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ScalarSourceRef<'a> {
    Fixed {
        value: f64,
    },
    Uniform {
        min: f64,
        max: f64,
    },
    Histogram {
        edges: &'a [f64],
        weights: &'a [f64],
    },
}

#[derive(Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ScalarSourceOwned {
    /// Use one value.
    Fixed {
        /// Fixed scalar value.
        value: f64,
    },
    /// Draw uniformly from `[min, max)`.
    Uniform {
        /// Lower bound.
        min: f64,
        /// Upper bound.
        max: f64,
    },
    /// Draw from an inline piecewise-constant histogram.
    Histogram {
        /// Bin edges, with one more edge than weight.
        edges: Vec<f64>,
        /// Nonnegative bin weights.
        weights: Vec<f64>,
    },
}

impl JsonSchema for ScalarSource {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ScalarSource".into()
    }

    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        ScalarSourceOwned::json_schema(generator)
    }
}

impl Serialize for ScalarSource {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Constant(value) => ScalarSourceRef::Fixed { value: *value },
            Self::Uniform { low, high } => ScalarSourceRef::Uniform {
                min: *low,
                max: *high,
            },
            Self::Histogram(histogram) => ScalarSourceRef::Histogram {
                edges: histogram.bin_edges(),
                weights: histogram.counts(),
            },
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ScalarSource {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match ScalarSourceOwned::deserialize(deserializer)? {
            ScalarSourceOwned::Fixed { value } => Ok(Self::constant(value)),
            ScalarSourceOwned::Uniform { min, max } => Ok(Self::uniform(min, max)),
            ScalarSourceOwned::Histogram { edges, weights } => Histogram::new(weights, edges)
                .map(Self::histogram)
                .map_err(serde::de::Error::custom),
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Scalar draw and its inverse proposal-density correction.
pub struct ScalarProposalResult {
    /// Sampled scalar value.
    pub value: f64,
    /// The proposal correction `1 / q(value)`; constants use one.
    pub weight: f64,
}

impl ScalarSource {
    /// Construct a constant scalar source.
    pub fn constant(value: f64) -> Self {
        Self::Constant(value)
    }

    /// Construct a uniform scalar source.
    pub fn uniform(low: f64, high: f64) -> Self {
        Self::Uniform { low, high }
    }

    /// Construct a histogram-backed scalar source.
    pub fn histogram(histogram: Histogram) -> Self {
        Self::Histogram(histogram)
    }

    /// Validate the source and return the smallest and largest values in its support.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when a constant or bound is non-finite, a
    /// uniform interval is empty, or histogram weights are invalid.
    pub fn support(&self) -> LadduPhysicsResult<(f64, f64)> {
        match self {
            Self::Constant(value) if value.is_finite() => Ok((*value, *value)),
            Self::Constant(value) => Err(LadduPhysicsError::invalid_value(
                "constant scalar source",
                "finite",
                value,
            )),
            Self::Uniform { low, high } if low.is_finite() && high.is_finite() && high > low => {
                Ok((*low, *high))
            }
            Self::Uniform { low, high } => Err(LadduPhysicsError::invalid_relation(format!(
                "uniform scalar source requires finite low < high, got [{low}, {high}]"
            ))),
            Self::Histogram(histogram) => {
                Self::histogram_density(histogram).map(|density| density.support())
            }
        }
    }

    /// Draw a value and inverse-density weight from the source.
    ///
    /// # Errors
    ///
    /// Returns [`LadduPhysicsError`] when the source parameters or histogram
    /// weights are invalid, or a histogram sample cannot be assigned to a bin.
    pub fn sample(&self, rng: &mut ProposalRng) -> LadduPhysicsResult<ScalarProposalResult> {
        match self {
            Self::Constant(value) if value.is_finite() => Ok(ScalarProposalResult {
                value: *value,
                weight: 1.0,
            }),
            Self::Constant(value) => Err(LadduPhysicsError::invalid_value(
                "constant scalar source",
                "finite",
                value,
            )),
            Self::Uniform { low, high } if low.is_finite() && high.is_finite() && high > low => {
                Ok(ScalarProposalResult {
                    value: low + rng.uniform() * (high - low),
                    weight: high - low,
                })
            }
            Self::Uniform { low, high } => Err(LadduPhysicsError::invalid_relation(format!(
                "uniform scalar source requires finite low < high, got [{low}, {high}]"
            ))),
            Self::Histogram(histogram) => {
                let mut histogram_rng = fastrand::Rng::with_seed(rng.next_u64());
                let value = histogram.sample(&mut histogram_rng)?;
                histogram.bin_index(value).ok_or_else(|| {
                    LadduPhysicsError::invalid_relation(
                        "sampled histogram value does not belong to an in-range bin",
                    )
                })?;
                let probability_density = Self::histogram_density(histogram)?.density(
                    histogram.bin_edges()[0],
                    histogram.bin_edges()[histogram.bin_edges().len() - 1],
                    value,
                );
                Ok(ScalarProposalResult {
                    value,
                    weight: probability_density.recip(),
                })
            }
        }
    }

    fn histogram_density(histogram: &Histogram) -> LadduPhysicsResult<PiecewiseDensity> {
        PiecewiseDensity::from_histogram(histogram).map_err(|_| {
            LadduPhysicsError::invalid_value(
                "histogram scalar-source counts",
                "finite and nonnegative with positive finite total weight",
                format!("{:?}", histogram.counts()),
            )
        })
    }
}
