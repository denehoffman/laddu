use laddu_generation::{
    ChannelGenerator, EnvelopeMode, EnvelopeOverflow, GenerationReport, ModelEvaluator,
    UnweightedConfig, WeightedConfig,
};
use laddu_physics::{
    generation::{
        InitialMomentum, MassProposal, ScalarSource, TComponent, TDistribution, VertexProposal,
    },
    vectors::{RealVec3, RealVec4},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyAny};

use super::{
    data::PyDataset,
    error::to_py_err,
    histogram::PyHistogram,
    model::{PyModel, model_free_values},
    runtime::{PyExecution, parse_memory_budget},
    topology::PyChannel,
};

#[pyclass(name = "MassProposal", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A fixed or uniformly sampled particle mass.
///
/// Parameters
/// ----------
/// value : float
///     Fixed mass, or the lower limit when ``high`` is provided.
/// high : float, optional
///     Exclusive upper limit of a uniform mass proposal.
///
/// Examples
/// --------
/// >>> import laddu as ld
/// >>> fixed = ld.generation.MassProposal(0.13957)
/// >>> broad = ld.generation.MassProposal(0.5, high=1.5)
pub struct PyMassProposal {
    pub(crate) inner: MassProposal,
}

#[pymethods]
impl PyMassProposal {
    /// Construct a mass proposal.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a limit is non-finite, a fixed mass is negative, or ``high`` is
    ///     not greater than ``value``.
    #[new]
    #[pyo3(signature = (value, *, high=None))]
    fn new(value: f64, high: Option<f64>) -> PyResult<Self> {
        if !value.is_finite() || high.is_some_and(|high| !high.is_finite()) {
            return Err(PyValueError::new_err("mass limits must be finite"));
        }
        let inner = match high {
            Some(high) if high > value => MassProposal::uniform(value, high),
            Some(_) => return Err(PyValueError::new_err("uniform mass requires low < high")),
            None if value >= 0.0 => MassProposal::fixed(value),
            None => return Err(PyValueError::new_err("fixed mass must be non-negative")),
        };
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!("MassProposal({:?})", self.inner)
    }
}

#[pyclass(
    name = "InitialMomentum",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// A fixed or sampled initial-state momentum prescription.
///
/// Use the static constructors to specify a complete four-vector, a
/// three-momentum with energy inferred from mass, or an energy distribution
/// along a fixed direction.
pub struct PyInitialMomentum {
    pub(crate) inner: InitialMomentum,
}

fn real_vec3(values: [f64; 3]) -> RealVec3 {
    RealVec3::new(values[0], values[1], values[2])
}

#[pymethods]
impl PyInitialMomentum {
    #[staticmethod]
    /// Use a fixed four-vector in ``(E, px, py, pz)`` order.
    ///
    /// Parameters
    /// ----------
    /// values : sequence of 4 float
    ///     Four-momentum components.
    fn p4(values: [f64; 4]) -> Self {
        Self {
            inner: InitialMomentum::p4(RealVec4::new(values[0], values[1], values[2], values[3])),
        }
    }

    #[staticmethod]
    /// Use a fixed three-momentum and infer energy from the particle mass.
    ///
    /// Parameters
    /// ----------
    /// values : sequence of 3 float
    ///     Cartesian momentum components.
    fn momentum(values: [f64; 3]) -> Self {
        Self {
            inner: InitialMomentum::momentum(real_vec3(values)),
        }
    }

    #[staticmethod]
    /// Use a fixed energy along a direction.
    ///
    /// Parameters
    /// ----------
    /// energy : float
    ///     Initial-state energy.
    /// direction : sequence of 3 float
    ///     Direction vector, normalized internally.
    #[pyo3(signature = (energy, *, direction))]
    fn energy(energy: f64, direction: [f64; 3]) -> Self {
        Self {
            inner: InitialMomentum::energy_direction(energy, real_vec3(direction)),
        }
    }

    #[staticmethod]
    /// Sample energy uniformly along a fixed direction.
    ///
    /// Parameters
    /// ----------
    /// low, high : float
    ///     Uniform energy limits with ``low < high``.
    /// direction : sequence of 3 float
    ///     Direction vector.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the energy interval is non-finite or empty.
    #[pyo3(signature = (*, low, high, direction))]
    fn uniform_energy(low: f64, high: f64, direction: [f64; 3]) -> PyResult<Self> {
        if !low.is_finite() || !high.is_finite() || high <= low {
            return Err(PyValueError::new_err(
                "uniform energy requires finite low < high",
            ));
        }
        Ok(Self {
            inner: InitialMomentum::energy_source_direction(
                ScalarSource::uniform(low, high),
                real_vec3(direction),
            ),
        })
    }

    #[staticmethod]
    /// Sample energy from a histogram along a fixed direction.
    ///
    /// Parameters
    /// ----------
    /// histogram : Histogram
    ///     Non-negative energy distribution.
    /// direction : sequence of 3 float
    ///     Direction vector.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If the histogram has no valid sampling support.
    #[pyo3(signature = (histogram, *, direction))]
    fn histogram_energy(histogram: &PyHistogram, direction: [f64; 3]) -> PyResult<Self> {
        let source = ScalarSource::histogram(histogram.inner.clone());
        source.support().map_err(to_py_err)?;
        Ok(Self {
            inner: InitialMomentum::energy_source_direction(source, real_vec3(direction)),
        })
    }
}

#[pyclass(name = "VertexProposal", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// A phase-space proposal distribution for one channel vertex.
pub struct PyVertexProposal {
    pub(crate) inner: VertexProposal,
}

#[pymethods]
impl PyVertexProposal {
    #[staticmethod]
    /// Create an isotropic decay proposal.
    fn isotropic() -> Self {
        Self {
            inner: VertexProposal::isotropic_decay(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*, incoming, outgoing, slope=None, uniform_fraction=0.0, t_min=None, t_max=None))]
    /// Create a uniform, exponential, or mixed t-exchange proposal.
    ///
    /// Parameters
    /// ----------
    /// incoming, outgoing : str
    ///     Edge names defining the momentum transfer.
    /// slope : float, optional
    ///     Exponential slope. Omission selects a uniform distribution.
    /// uniform_fraction : float, default=0.0
    ///     Fraction mixed from a uniform component.
    /// t_min, t_max : float, optional
    ///     Optional momentum-transfer limits.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the mixture fraction or slope is invalid.
    /// LadduError
    ///     If the transfer limits do not form a valid interval.
    fn t_exchange(
        incoming: String,
        outgoing: String,
        slope: Option<f64>,
        uniform_fraction: f64,
        t_min: Option<f64>,
        t_max: Option<f64>,
    ) -> PyResult<Self> {
        if !uniform_fraction.is_finite() || !(0.0..=1.0).contains(&uniform_fraction) {
            return Err(PyValueError::new_err(
                "uniform_fraction must be between 0 and 1",
            ));
        }
        if slope.is_some_and(|slope| !slope.is_finite()) {
            return Err(PyValueError::new_err("t slope must be finite"));
        }
        let distribution = match slope {
            Some(_) if uniform_fraction == 1.0 => TDistribution::uniform(),
            Some(slope) if uniform_fraction > 0.0 => TDistribution::mixture([
                (uniform_fraction, TComponent::Uniform),
                (1.0 - uniform_fraction, TComponent::Exponential { slope }),
            ]),
            Some(slope) => TDistribution::exponential(slope),
            None => TDistribution::uniform(),
        }
        .with_limits(t_min, t_max)
        .map_err(to_py_err)?;
        Ok(Self {
            inner: VertexProposal::t_exchange((incoming, outgoing), distribution),
        })
    }

    #[staticmethod]
    /// Create a pole-shaped t-exchange proposal.
    ///
    /// Parameters
    /// ----------
    /// incoming, outgoing : str
    ///     Edge names defining the momentum transfer.
    /// exchange_mass : float
    ///     Non-negative pole mass.
    /// power : float
    ///     Positive pole power.
    #[pyo3(signature = (*, incoming, outgoing, exchange_mass, power))]
    fn t_exchange_pole(
        incoming: String,
        outgoing: String,
        exchange_mass: f64,
        power: f64,
    ) -> PyResult<Self> {
        if !exchange_mass.is_finite() || exchange_mass < 0.0 || !power.is_finite() || power <= 0.0 {
            return Err(PyValueError::new_err(
                "pole exchange_mass must be finite and non-negative and power must be finite and positive",
            ));
        }
        Ok(Self {
            inner: VertexProposal::t_exchange(
                (incoming, outgoing),
                TDistribution::pole(exchange_mass, power),
            ),
        })
    }

    #[staticmethod]
    /// Create a histogram-sampled t-exchange proposal.
    ///
    /// Parameters
    /// ----------
    /// incoming, outgoing : str
    ///     Edge names defining the momentum transfer.
    /// histogram : Histogram
    ///     Non-negative transfer distribution with positive total weight.
    #[pyo3(signature = (*, incoming, outgoing, histogram))]
    fn t_exchange_histogram(
        incoming: String,
        outgoing: String,
        histogram: &PyHistogram,
    ) -> PyResult<Self> {
        if histogram.inner.total_weight() <= 0.0
            || histogram.inner.counts().iter().any(|count| *count < 0.0)
        {
            return Err(PyValueError::new_err(
                "t-exchange histogram counts must be non-negative with positive total weight",
            ));
        }
        Ok(Self {
            inner: VertexProposal::t_exchange(
                (incoming, outgoing),
                TDistribution::histogram(histogram.inner.clone()),
            ),
        })
    }
}

#[pyclass(
    name = "GenerationReport",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Diagnostics from a weighted or unweighted generation run.
///
/// Attributes
/// ----------
/// requested, produced, proposals, rejected : int
///     Event and proposal counts.
/// acceptance_rate : float
///     Produced events divided by proposals.
/// envelope : float or None
///     Unweighting envelope, when applicable.
/// maximum_weight, minimum_weight, sum_weights : float
///     Observed weight statistics.
/// seed : int
///     Random seed used for generation.
pub struct PyGenerationReport {
    inner: GenerationReport,
}

#[pymethods]
impl PyGenerationReport {
    #[getter]
    /// int: Number of requested output events.
    fn requested(&self) -> usize {
        self.inner.requested
    }
    #[getter]
    /// int: Number of events successfully produced.
    fn produced(&self) -> usize {
        self.inner.produced
    }
    #[getter]
    /// int: Number of phase-space proposals attempted.
    fn proposals(&self) -> usize {
        self.inner.proposals
    }
    #[getter]
    /// int: Number of rejected proposals.
    fn rejected(&self) -> usize {
        self.inner.rejected
    }
    #[getter]
    /// float: Fraction of proposals accepted.
    fn acceptance_rate(&self) -> f64 {
        self.inner.acceptance_rate()
    }
    #[getter]
    /// float or None: Unweighting envelope used by the run.
    fn envelope(&self) -> Option<f64> {
        self.inner.envelope
    }
    #[getter]
    /// float: Largest generated model weight.
    fn maximum_weight(&self) -> f64 {
        self.inner.maximum_weight
    }
    #[getter]
    /// float: Smallest generated model weight.
    fn minimum_weight(&self) -> f64 {
        self.inner.minimum_weight
    }
    #[getter]
    /// float: Sum of generated model weights.
    fn sum_weights(&self) -> f64 {
        self.inner.sum_weights
    }
    #[getter]
    /// int: Random seed used by the run.
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    /// int: Event count selected for each generation chunk.
    fn chunk_events(&self) -> usize {
        self.inner.chunk_events
    }
    #[getter]
    /// int: Planned peak memory use in bytes.
    fn estimated_peak_bytes(&self) -> u64 {
        self.inner.estimated_peak_bytes
    }
    #[getter]
    /// int or None: Observed allocator high-water mark when available.
    fn actual_high_water_bytes(&self) -> Option<u64> {
        self.inner.actual_high_water_bytes
    }
    fn __repr__(&self) -> String {
        format!(
            "GenerationReport(produced={}, proposals={}, acceptance_rate={:.3})",
            self.inner.produced,
            self.inner.proposals,
            self.inner.acceptance_rate()
        )
    }
}

#[pyclass(name = "Generator", module = "laddu", frozen, skip_from_py_object)]
/// A phase-space event generator compiled from a reaction channel.
///
/// Parameters
/// ----------
/// channel : Channel
///     Fully specified reaction topology and proposal configuration.
pub struct PyGenerator {
    inner: ChannelGenerator,
}

fn model_values(
    model: &PyModel,
    parameters: Option<&Bound<'_, PyAny>>,
) -> PyResult<laddu_expr::parameters::ParamValues> {
    let values = match parameters {
        None => model.inner.params().initial_free_values(),
        Some(values) => model_free_values(&model.inner, values)?,
    };
    model.inner.params().values(&values).map_err(to_py_err)
}

#[pymethods]
impl PyGenerator {
    /// Compile a channel into an event generator.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If masses, momenta, vertices, or proposals are incomplete.
    #[new]
    fn new(channel: &PyChannel) -> PyResult<Self> {
        Ok(Self {
            inner: ChannelGenerator::new(channel.inner.clone()).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (
        events,
        *,
        model=None,
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        execution=None,
        memory: "MemoryBudget | int | str | None" = None,
        seed=0,
        diagnostics=false
    ))]
    #[allow(clippy::too_many_arguments)]
    /// Generate weighted phase-space events.
    ///
    /// Parameters
    /// ----------
    /// events : int
    ///     Number of events to produce.
    /// model : Model, optional
    ///     Model used as the event weight. Unit weights are used when omitted.
    /// parameters : sequence of float or dict, optional
    ///     Model parameter values.
    /// execution : Execution, optional
    ///     Runtime used to evaluate ``model``.
    /// memory : MemoryBudget, int, or str, optional
    ///     Host-memory budget used to choose the proposal chunk size.
    /// seed : int, default=0
    ///     Random seed.
    /// diagnostics : bool, default=False
    ///     Collect additional generation diagnostics.
    ///
    /// Returns
    /// -------
    /// dataset : Dataset
    ///     Generated events with model weights.
    /// report : GenerationReport
    ///     Counts and weight statistics.
    fn weighted(
        &self,
        py: Python<'_>,
        events: usize,
        model: Option<&PyModel>,
        parameters: Option<&Bound<'_, PyAny>>,
        execution: Option<&PyExecution>,
        memory: Option<&Bound<'_, PyAny>>,
        seed: u64,
        diagnostics: bool,
    ) -> PyResult<(PyDataset, PyGenerationReport)> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        let evaluator = model
            .map(|model| {
                ModelEvaluator::prepare(
                    &model.inner,
                    model_values(model, parameters)?,
                    &execution.inner,
                )
                .map_err(to_py_err)
            })
            .transpose()?;
        let config = WeightedConfig {
            events,
            memory: memory
                .map(parse_memory_budget)
                .transpose()?
                .unwrap_or(laddu_runtime::MemoryBudget::Auto),
            seed,
            diagnostics,
        };
        let (dataset, report) = py
            .detach(|| {
                self.inner
                    .generate_weighted_dataset(config, evaluator.as_ref())
            })
            .map_err(to_py_err)?;
        Ok((
            PyDataset { inner: dataset },
            PyGenerationReport { inner: report },
        ))
    }

    #[pyo3(signature = (
        events,
        model,
        *,
        parameters: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        execution=None,
        memory: "MemoryBudget | int | str | None" = None,
        seed=0,
        max_proposals=None,
        max_weight=None,
        pilot_proposals=10_000,
        safety_factor=2.0,
        grow_envelope=false,
        diagnostics=false
    ))]
    #[allow(clippy::too_many_arguments)]
    /// Generate accept-reject unweighted events.
    ///
    /// Parameters
    /// ----------
    /// events : int
    ///     Number of accepted events to produce.
    /// model : Model
    ///     Non-negative event intensity used for accept-reject sampling.
    /// parameters : sequence of float or dict, optional
    ///     Model parameter values.
    /// execution : Execution, optional
    ///     Runtime used to evaluate ``model``.
    /// memory : MemoryBudget, int, or str, optional
    ///     Host-memory budget used to choose the proposal chunk size.
    /// seed : int, default=0
    ///     Random seed.
    /// max_proposals : int, optional
    ///     Stop with an error after this many proposals.
    /// max_weight : float, optional
    ///     Strict known envelope. If omitted, estimate one with a pilot run.
    /// pilot_proposals : int, default=10000
    ///     Number of proposals used for envelope estimation.
    /// safety_factor : float, default=2.0
    ///     Multiplier applied to pilot or grown envelopes.
    /// grow_envelope : bool, default=False
    ///     Grow an underestimated envelope instead of raising an error.
    /// diagnostics : bool, default=False
    ///     Collect additional generation diagnostics.
    ///
    /// Returns
    /// -------
    /// dataset : Dataset
    ///     Accepted unit-weight events.
    /// report : GenerationReport
    ///     Acceptance and envelope statistics.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If model evaluation fails, the envelope is exceeded in strict mode,
    ///     or the proposal limit is reached.
    fn unweighted(
        &self,
        py: Python<'_>,
        events: usize,
        model: &PyModel,
        parameters: Option<&Bound<'_, PyAny>>,
        execution: Option<&PyExecution>,
        memory: Option<&Bound<'_, PyAny>>,
        seed: u64,
        max_proposals: Option<usize>,
        max_weight: Option<f64>,
        pilot_proposals: usize,
        safety_factor: f64,
        grow_envelope: bool,
        diagnostics: bool,
    ) -> PyResult<(PyDataset, PyGenerationReport)> {
        let execution = execution
            .cloned()
            .map(Ok)
            .unwrap_or_else(PyExecution::default_inner)?;
        let evaluator = ModelEvaluator::prepare(
            &model.inner,
            model_values(model, parameters)?,
            &execution.inner,
        )
        .map_err(to_py_err)?;
        let envelope = match max_weight {
            Some(max_weight) => EnvelopeMode::Strict { max_weight },
            None => EnvelopeMode::Pilot {
                proposals: pilot_proposals,
                safety_factor,
            },
        };
        let envelope_overflow = if grow_envelope {
            EnvelopeOverflow::Grow { safety_factor }
        } else {
            EnvelopeOverflow::Error
        };
        let config = UnweightedConfig {
            events,
            max_proposals,
            memory: memory
                .map(parse_memory_budget)
                .transpose()?
                .unwrap_or(laddu_runtime::MemoryBudget::Auto),
            seed,
            diagnostics,
            envelope,
            envelope_overflow,
        };
        let (dataset, report) = py
            .detach(|| self.inner.generate_unweighted_dataset(config, &evaluator))
            .map_err(to_py_err)?;
        Ok((
            PyDataset { inner: dataset },
            PyGenerationReport { inner: report },
        ))
    }
}

#[pymodule(submodule)]
/// Phase-space proposals, event generators, and generation diagnostics.
pub mod generation {
    #[pymodule_export]
    use super::{
        PyGenerationReport as GenerationReport, PyGenerator as Generator,
        PyInitialMomentum as InitialMomentum, PyMassProposal as MassProposal,
        PyVertexProposal as VertexProposal,
    };
}
