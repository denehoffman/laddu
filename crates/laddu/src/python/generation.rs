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
    runtime::PyExecution,
    topology::PyChannel,
};

#[pyclass(name = "MassProposal", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyMassProposal {
    pub(crate) inner: MassProposal,
}

#[pymethods]
impl PyMassProposal {
    #[new]
    #[pyo3(signature = (value, high=None))]
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
pub struct PyInitialMomentum {
    pub(crate) inner: InitialMomentum,
}

fn real_vec3(values: [f64; 3]) -> RealVec3 {
    RealVec3::new(values[0], values[1], values[2])
}

#[pymethods]
impl PyInitialMomentum {
    #[staticmethod]
    fn p4(values: [f64; 4]) -> Self {
        Self {
            inner: InitialMomentum::p4(RealVec4::new(values[0], values[1], values[2], values[3])),
        }
    }

    #[staticmethod]
    fn momentum(values: [f64; 3]) -> Self {
        Self {
            inner: InitialMomentum::momentum(real_vec3(values)),
        }
    }

    #[staticmethod]
    fn energy(energy: f64, direction: [f64; 3]) -> Self {
        Self {
            inner: InitialMomentum::energy_direction(energy, real_vec3(direction)),
        }
    }

    #[staticmethod]
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
pub struct PyVertexProposal {
    pub(crate) inner: VertexProposal,
}

#[pymethods]
impl PyVertexProposal {
    #[staticmethod]
    fn isotropic() -> Self {
        Self {
            inner: VertexProposal::isotropic_decay(),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (incoming, outgoing, *, slope=None, uniform_fraction=0.0, t_min=None, t_max=None))]
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
pub struct PyGenerationReport {
    inner: GenerationReport,
}

#[pymethods]
impl PyGenerationReport {
    #[getter]
    fn requested(&self) -> usize {
        self.inner.requested
    }
    #[getter]
    fn produced(&self) -> usize {
        self.inner.produced
    }
    #[getter]
    fn proposals(&self) -> usize {
        self.inner.proposals
    }
    #[getter]
    fn rejected(&self) -> usize {
        self.inner.rejected
    }
    #[getter]
    fn acceptance_rate(&self) -> f64 {
        self.inner.acceptance_rate()
    }
    #[getter]
    fn envelope(&self) -> Option<f64> {
        self.inner.envelope
    }
    #[getter]
    fn maximum_weight(&self) -> f64 {
        self.inner.maximum_weight
    }
    #[getter]
    fn minimum_weight(&self) -> f64 {
        self.inner.minimum_weight
    }
    #[getter]
    fn sum_weights(&self) -> f64 {
        self.inner.sum_weights
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
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
    #[new]
    fn new(channel: &PyChannel) -> PyResult<Self> {
        Ok(Self {
            inner: ChannelGenerator::new(channel.inner.clone()).map_err(to_py_err)?,
        })
    }

    #[pyo3(signature = (events, *, model=None, parameters=None, execution=None, batch_size=1024, seed=0, diagnostics=false))]
    #[allow(clippy::too_many_arguments)]
    fn weighted(
        &self,
        py: Python<'_>,
        events: usize,
        model: Option<&PyModel>,
        parameters: Option<&Bound<'_, PyAny>>,
        execution: Option<&PyExecution>,
        batch_size: usize,
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
            batch_size,
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

    #[pyo3(signature = (events, model, *, parameters=None, execution=None, batch_size=1024, seed=0, max_proposals=None, max_weight=None, pilot_proposals=10_000, safety_factor=2.0, grow_envelope=false, diagnostics=false))]
    #[allow(clippy::too_many_arguments)]
    fn unweighted(
        &self,
        py: Python<'_>,
        events: usize,
        model: &PyModel,
        parameters: Option<&Bound<'_, PyAny>>,
        execution: Option<&PyExecution>,
        batch_size: usize,
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
            batch_size,
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
pub mod generation {
    #[pymodule_export]
    use super::{
        PyGenerationReport as GenerationReport, PyGenerator as Generator,
        PyInitialMomentum as InitialMomentum, PyMassProposal as MassProposal,
        PyVertexProposal as VertexProposal,
    };
}
