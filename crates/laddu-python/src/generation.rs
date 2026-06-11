use std::{cell::RefCell, rc::Rc, sync::Arc};

use laddu_core::{MassSampler, MomentumSource, VertexGenerator};
use laddu_generation::{
    gen, DatasetSink, DecayParticlePlan, DecayPlan, EventGenerator, GeneratedEvent,
    GenerationMode as RustGenerationMode, GenerationOptions, GenerationOutput, GenerationPlan,
    GenerationResult, GenerationStats, InitialParticlePlan, PlannedMass, ProductionPlan,
};
use pyo3::prelude::*;

use crate::{
    amplitudes::PyExpression, data::PyDataset, math::PyHistogram, variables::PyChannel,
    vectors::PyVec4,
};

#[pyclass(name = "MomentumSource", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyMomentumSource(pub MomentumSource);

#[pymethods]
impl PyMomentumSource {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "MassSampler", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyMassSampler(pub MassSampler);

#[pymethods]
impl PyMassSampler {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "VertexGenerator", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyVertexGenerator(pub VertexGenerator);

#[pymethods]
impl PyVertexGenerator {
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "GenerationMode", module = "laddu", from_py_object)]
#[derive(Clone, Default)]
pub struct PyGenerationMode(pub RustGenerationMode);

#[pymethods]
impl PyGenerationMode {
    #[staticmethod]
    fn raw() -> Self {
        Self(RustGenerationMode::Raw)
    }

    #[staticmethod]
    fn weighted(expression: &PyExpression, parameters: Vec<f64>) -> Self {
        Self(RustGenerationMode::Weighted {
            expression: Box::new(expression.0.clone()),
            parameters,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "GenerationOptions", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyGenerationOptions(pub GenerationOptions);

#[pymethods]
impl PyGenerationOptions {
    #[new]
    #[pyo3(signature=(*, batch_size=10_000, max_trials=None, seed=None))]
    fn new(batch_size: usize, max_trials: Option<u64>, seed: Option<u64>) -> Self {
        Self(GenerationOptions {
            batch_size,
            max_trials,
            seed,
        })
    }

    #[getter]
    fn batch_size(&self) -> usize {
        self.0.batch_size
    }

    #[getter]
    fn max_trials(&self) -> Option<u64> {
        self.0.max_trials
    }

    #[getter]
    fn seed(&self) -> Option<u64> {
        self.0.seed
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "GenerationOutput", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyGenerationOutput(pub GenerationOutput);

#[pymethods]
impl PyGenerationOutput {
    #[staticmethod]
    fn all() -> Self {
        Self(GenerationOutput::all())
    }

    #[staticmethod]
    fn final_state() -> Self {
        Self(GenerationOutput::final_state())
    }

    #[staticmethod]
    fn only(labels: Vec<String>) -> Self {
        Self(GenerationOutput::only(labels))
    }

    #[staticmethod]
    fn exclude(labels: Vec<String>) -> Self {
        Self(GenerationOutput::exclude(labels))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "DatasetSink", module = "laddu", from_py_object)]
#[derive(Clone, Default)]
pub struct PyDatasetSink(pub DatasetSink);

#[pymethods]
impl PyDatasetSink {
    #[new]
    #[pyo3(signature=(*, output=None))]
    fn new(output: Option<&PyGenerationOutput>) -> Self {
        let sink = output.map_or_else(DatasetSink::new, |output| {
            DatasetSink::new().output(output.0.clone())
        });
        Self(sink)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "GenerationStats", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGenerationStats(pub GenerationStats);

#[pymethods]
impl PyGenerationStats {
    #[getter]
    fn target_events(&self) -> u64 {
        self.0.target_events
    }

    #[getter]
    fn written_events(&self) -> u64 {
        self.0.written_events
    }

    #[getter]
    fn proposed_events(&self) -> u64 {
        self.0.proposed_events
    }

    #[getter]
    fn accepted_events(&self) -> u64 {
        self.0.accepted_events
    }

    #[getter]
    fn rejected_events(&self) -> u64 {
        self.0.rejected_events
    }

    #[getter]
    fn acceptance_rate(&self) -> Option<f64> {
        self.0.acceptance_rate
    }

    #[getter]
    fn envelope(&self) -> Option<f64> {
        self.0.envelope
    }

    #[getter]
    fn envelope_violations(&self) -> u64 {
        self.0.envelope_violations
    }

    #[getter]
    fn sum_weights(&self) -> f64 {
        self.0.sum_weights
    }

    #[getter]
    fn min_weight(&self) -> Option<f64> {
        self.0.min_weight
    }

    #[getter]
    fn max_weight(&self) -> Option<f64> {
        self.0.max_weight
    }

    #[getter]
    fn batches_written(&self) -> u64 {
        self.0.batches_written
    }

    fn audit(&self) -> String {
        self.0.audit()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "GenerationResult", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGenerationResult(pub GenerationResult<laddu_core::Dataset>);

impl From<GenerationResult<laddu_core::Dataset>> for PyGenerationResult {
    fn from(result: GenerationResult<laddu_core::Dataset>) -> Self {
        Self(result)
    }
}

#[pymethods]
impl PyGenerationResult {
    #[getter]
    fn output(&self) -> PyDataset {
        PyDataset(Arc::new(self.0.output.clone()))
    }

    #[getter]
    fn stats(&self) -> PyGenerationStats {
        PyGenerationStats(self.0.stats.clone())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0.stats)
    }
}

#[pyclass(name = "PlannedMass", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyPlannedMass(pub PlannedMass);

#[pymethods]
impl PyPlannedMass {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.0 {
            PlannedMass::Properties(_) => "properties",
            PlannedMass::Sampled(_) => "sampled",
        }
    }

    #[getter]
    fn value(&self) -> Option<f64> {
        match self.0 {
            PlannedMass::Properties(value) => Some(value),
            PlannedMass::Sampled(_) => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "InitialParticlePlan", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyInitialParticlePlan(pub InitialParticlePlan);

#[pymethods]
impl PyInitialParticlePlan {
    #[getter]
    fn label(&self) -> String {
        self.0.label().to_string()
    }

    #[getter]
    fn mass(&self) -> f64 {
        self.0.mass()
    }

    #[getter]
    fn momentum(&self) -> PyMomentumSource {
        PyMomentumSource(self.0.momentum().clone())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "DecayParticlePlan", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDecayParticlePlan(pub DecayParticlePlan);

#[pymethods]
impl PyDecayParticlePlan {
    #[getter]
    fn label(&self) -> String {
        self.0.label().to_string()
    }

    #[getter]
    fn mass(&self) -> PyPlannedMass {
        PyPlannedMass(self.0.mass().clone())
    }

    #[getter]
    fn decay(&self) -> Option<PyDecayPlan> {
        self.0.decay().cloned().map(PyDecayPlan)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "DecayPlan", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDecayPlan(pub DecayPlan);

#[pymethods]
impl PyDecayPlan {
    #[getter]
    fn vertex(&self) -> String {
        self.0.vertex().to_string()
    }

    #[getter]
    fn daughters(&self) -> Vec<PyDecayParticlePlan> {
        self.0
            .daughters()
            .iter()
            .cloned()
            .map(PyDecayParticlePlan)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "ProductionPlan", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyProductionPlan(pub ProductionPlan);

#[pymethods]
impl PyProductionPlan {
    #[getter]
    fn vertex(&self) -> String {
        self.0.vertex().to_string()
    }

    #[getter]
    fn incoming(&self) -> Vec<PyInitialParticlePlan> {
        self.0
            .incoming()
            .iter()
            .cloned()
            .map(PyInitialParticlePlan)
            .collect()
    }

    #[getter]
    fn outgoing(&self) -> Vec<PyDecayParticlePlan> {
        self.0
            .outgoing()
            .iter()
            .cloned()
            .map(PyDecayParticlePlan)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "GenerationPlan", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGenerationPlan(pub GenerationPlan);

#[pymethods]
impl PyGenerationPlan {
    #[new]
    fn new(channel: &PyChannel) -> PyResult<Self> {
        Ok(Self(GenerationPlan::from_channel(&channel.channel())?))
    }

    #[staticmethod]
    fn from_channel(channel: &PyChannel) -> PyResult<Self> {
        Self::new(channel)
    }

    #[getter]
    fn production(&self) -> PyProductionPlan {
        PyProductionPlan(self.0.production().clone())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "GeneratedEvent", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGeneratedEvent(pub GeneratedEvent);

#[pymethods]
impl PyGeneratedEvent {
    fn labels(&self) -> Vec<String> {
        self.0.labels().map(ToString::to_string).collect()
    }

    fn p4(&self, label: &str) -> Option<PyVec4> {
        self.0.p4(label).map(PyVec4)
    }

    fn p4s(&self) -> Vec<(String, PyVec4)> {
        self.0
            .p4s()
            .iter()
            .map(|(label, p4)| (label.clone(), PyVec4(*p4)))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(
    name = "EventGenerator",
    module = "laddu",
    skip_from_py_object,
    unsendable
)]
#[derive(Clone)]
pub struct PyEventGenerator {
    generator: EventGenerator,
    rng: Rc<RefCell<fastrand::Rng>>,
}

impl PyEventGenerator {
    fn new_inner(generator: EventGenerator, seed: Option<u64>) -> Self {
        let rng = seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
        Self {
            generator,
            rng: Rc::new(RefCell::new(rng)),
        }
    }
}

#[pymethods]
impl PyEventGenerator {
    #[new]
    #[pyo3(signature = (channel, *, seed=None))]
    fn new(channel: &PyChannel, seed: Option<u64>) -> PyResult<Self> {
        let generator = EventGenerator::from_channel(&channel.channel())?;
        let generator = match seed {
            Some(seed) => generator.with_seed(seed),
            None => generator,
        };
        Ok(Self::new_inner(generator, seed))
    }

    #[staticmethod]
    #[pyo3(signature = (channel, *, seed=None))]
    fn from_channel(channel: &PyChannel, seed: Option<u64>) -> PyResult<Self> {
        Self::new(channel, seed)
    }

    fn with_seed(&self, seed: u64) -> Self {
        Self::new_inner(self.generator.clone().with_seed(seed), Some(seed))
    }

    #[getter]
    fn plan(&self) -> PyGenerationPlan {
        PyGenerationPlan(self.generator.plan().clone())
    }

    fn p4_labels(&self) -> Vec<String> {
        self.generator.p4_labels().to_vec()
    }

    fn generate_event(&self) -> PyResult<PyGeneratedEvent> {
        Ok(PyGeneratedEvent(
            self.generator.generate_event(&mut self.rng.borrow_mut())?,
        ))
    }

    #[pyo3(signature=(target_events, sink, *, mode=None, options=None))]
    fn generate(
        &self,
        target_events: usize,
        sink: &PyDatasetSink,
        mode: Option<&PyGenerationMode>,
        options: Option<&PyGenerationOptions>,
    ) -> PyResult<PyGenerationResult> {
        let mode = mode.map(|mode| mode.0.clone()).unwrap_or_default();
        let options = options.map(|options| options.0.clone()).unwrap_or_default();
        Ok(self
            .generator
            .generate(target_events, sink.0.clone(), mode, options)?
            .into())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.generator)
    }
}

#[pyfunction(name = "energy")]
pub fn py_energy(value: f64) -> PyMomentumSource {
    PyMomentumSource(gen::energy(value))
}

#[pyfunction(name = "uniform_energy")]
pub fn py_uniform_energy(min: f64, max: f64) -> PyMomentumSource {
    PyMomentumSource(gen::uniform_energy(min, max))
}

#[pyfunction(name = "histogram_energy")]
pub fn py_histogram_energy(histogram: &PyHistogram) -> PyResult<PyMomentumSource> {
    Ok(PyMomentumSource(gen::histogram_energy(
        histogram.0.clone(),
    )?))
}

#[pyfunction(name = "rest")]
pub fn py_rest() -> PyMomentumSource {
    PyMomentumSource(gen::rest())
}

#[pyfunction(name = "uniform_mass")]
pub fn py_uniform_mass(min: f64, max: f64) -> PyMassSampler {
    PyMassSampler(gen::uniform_mass(min, max))
}

#[pyfunction(name = "histogram_mass")]
pub fn py_histogram_mass(histogram: &PyHistogram) -> PyResult<PyMassSampler> {
    Ok(PyMassSampler(gen::histogram_mass(histogram.0.clone())?))
}

#[pyfunction(name = "mass_from_properties")]
pub fn py_mass_from_properties() -> PyMassSampler {
    PyMassSampler(MassSampler::FromProperties)
}

#[pyfunction(name = "t_exponential")]
pub fn py_t_exponential(slope: f64) -> PyVertexGenerator {
    PyVertexGenerator(gen::t_exponential(slope))
}

#[pyfunction(name = "t_histogram")]
pub fn py_t_histogram(histogram: &PyHistogram) -> PyResult<PyVertexGenerator> {
    Ok(PyVertexGenerator(gen::t_histogram(histogram.0.clone())?))
}
