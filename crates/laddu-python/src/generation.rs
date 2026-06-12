use std::{cell::RefCell, rc::Rc, sync::Arc};

use laddu_core::{LadduError, LadduResult, MassSampler, MomentumSource, VertexGenerator};
use laddu_generation::{
    gen, DatasetSink, DecayParticlePlan, DecayPlan, Envelope, EnvelopeStats,
    EnvelopeViolationPolicy, EventGenerator, GeneratedBatchView, GeneratedEvent, GeneratedLayout,
    GeneratedSink, GenerationMode as RustGenerationMode, GenerationOptions, GenerationOutput,
    GenerationPlan, GenerationResult, GenerationStats, InitialParticlePlan, PlannedMass,
    ProductionPlan, SinkMpiSupport,
};
use pyo3::{exceptions::PyTypeError, prelude::*, IntoPyObjectExt};

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

    #[staticmethod]
    #[pyo3(signature=(expression, parameters, *, envelope))]
    fn accepted(
        expression: &PyExpression,
        parameters: Vec<f64>,
        envelope: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self(RustGenerationMode::Accepted {
            expression: Box::new(expression.0.clone()),
            parameters,
            envelope: py_envelope_arg(envelope)?,
        }))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(name = "Envelope", module = "laddu", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyEnvelope(pub Envelope);

#[pymethods]
impl PyEnvelope {
    #[staticmethod]
    fn initial(value: f64) -> Self {
        Self(Envelope::initial(value))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

fn py_envelope_arg(arg: &Bound<'_, PyAny>) -> PyResult<Envelope> {
    if let Ok(envelope) = arg.extract::<PyEnvelope>() {
        return Ok(envelope.0);
    }
    if let Ok(value) = arg.extract::<f64>() {
        return Ok(Envelope::initial(value));
    }
    Err(PyTypeError::new_err(
        "expected envelope to be an Envelope or float",
    ))
}

#[pyclass(
    eq,
    eq_int,
    name = "EnvelopeViolationPolicy",
    module = "laddu",
    from_py_object
)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyEnvelopeViolationPolicy {
    Error = 0,
    WarnAndContinue = 1,
    Grow = 2,
}

impl From<PyEnvelopeViolationPolicy> for EnvelopeViolationPolicy {
    fn from(policy: PyEnvelopeViolationPolicy) -> Self {
        match policy {
            PyEnvelopeViolationPolicy::Error => Self::Error,
            PyEnvelopeViolationPolicy::WarnAndContinue => Self::WarnAndContinue,
            PyEnvelopeViolationPolicy::Grow => Self::Grow,
        }
    }
}

impl From<EnvelopeViolationPolicy> for PyEnvelopeViolationPolicy {
    fn from(policy: EnvelopeViolationPolicy) -> Self {
        match policy {
            EnvelopeViolationPolicy::Error => Self::Error,
            EnvelopeViolationPolicy::WarnAndContinue => Self::WarnAndContinue,
            EnvelopeViolationPolicy::Grow => Self::Grow,
        }
    }
}

#[pyclass(name = "GenerationOptions", module = "laddu", from_py_object)]
#[derive(Clone)]
pub struct PyGenerationOptions(pub GenerationOptions);

#[pymethods]
impl PyGenerationOptions {
    #[new]
    #[pyo3(signature=(*, batch_size=10_000, max_trials=None, seed=None, envelope_violation_policy=None))]
    fn new(
        batch_size: usize,
        max_trials: Option<u64>,
        seed: Option<u64>,
        envelope_violation_policy: Option<PyEnvelopeViolationPolicy>,
    ) -> Self {
        Self(GenerationOptions {
            batch_size,
            max_trials,
            seed,
            envelope_violation_policy: envelope_violation_policy
                .map_or(EnvelopeViolationPolicy::Error, Into::into),
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

    #[getter]
    fn envelope_violation_policy(&self) -> PyEnvelopeViolationPolicy {
        self.0.envelope_violation_policy.into()
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

/// Trait for native Rust-backed sinks exposed through Python.
pub trait PyGeneratedSinkImpl: 'static {
    /// Called once before records are pushed.
    fn begin(&mut self, _layout: &GeneratedLayout) -> LadduResult<()> {
        Ok(())
    }

    /// Push one batch of generated records.
    fn push_batch(&mut self, batch: GeneratedBatchView<'_>) -> LadduResult<()>;

    /// Finish output and return the Python-facing result object.
    fn finish(self: Box<Self>, py: Python<'_>) -> LadduResult<Py<PyAny>>;

    /// Return this sink's MPI output support.
    fn mpi_support(&self) -> SinkMpiSupport {
        SinkMpiSupport::RankLocal
    }
}

/// Type-erased native sink wrapper used by Python generation.
#[pyclass(
    name = "GeneratedSink",
    module = "laddu",
    skip_from_py_object,
    unsendable
)]
pub struct PyGeneratedSink {
    inner: Rc<RefCell<Option<Box<dyn PyGeneratedSinkImpl>>>>,
}

impl PyGeneratedSink {
    /// Wrap a native sink adapter so Python can pass it to `EventGenerator.generate`.
    pub fn new<S>(sink: S) -> Self
    where
        S: PyGeneratedSinkImpl,
    {
        Self {
            inner: Rc::new(RefCell::new(Some(Box::new(sink)))),
        }
    }
}

#[pymethods]
impl PyGeneratedSink {
    fn __repr__(&self) -> String {
        "GeneratedSink()".to_string()
    }
}

struct DatasetPySink {
    sink: DatasetSink,
}

impl PyGeneratedSinkImpl for DatasetPySink {
    fn begin(&mut self, layout: &GeneratedLayout) -> LadduResult<()> {
        self.sink.begin(layout)
    }

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>) -> LadduResult<()> {
        self.sink.push_batch(batch)
    }

    fn finish(self: Box<Self>, py: Python<'_>) -> LadduResult<Py<PyAny>> {
        PyDataset(Arc::new(self.sink.finish()?))
            .into_bound_py_any(py)
            .map(|object| object.unbind())
            .map_err(|err| LadduError::Custom(err.to_string()))
    }

    fn mpi_support(&self) -> SinkMpiSupport {
        self.sink.mpi_support()
    }
}

struct AnyPySink {
    inner: Rc<RefCell<Option<Box<dyn PyGeneratedSinkImpl>>>>,
}

impl GeneratedSink for AnyPySink {
    type Output = Py<PyAny>;

    fn begin(&mut self, layout: &GeneratedLayout) -> LadduResult<()> {
        let mut inner = self.inner.borrow_mut();
        inner
            .as_mut()
            .ok_or_else(|| LadduError::Custom("generated sink has already finished".to_string()))?
            .begin(layout)
    }

    fn push_batch(&mut self, batch: GeneratedBatchView<'_>) -> LadduResult<()> {
        let mut inner = self.inner.borrow_mut();
        inner
            .as_mut()
            .ok_or_else(|| LadduError::Custom("generated sink has already finished".to_string()))?
            .push_batch(batch)
    }

    fn finish(self) -> LadduResult<Self::Output> {
        let inner =
            self.inner.borrow_mut().take().ok_or_else(|| {
                LadduError::Custom("generated sink has already finished".to_string())
            })?;
        Python::attach(|py| inner.finish(py))
    }

    fn mpi_support(&self) -> SinkMpiSupport {
        let inner = self.inner.borrow();
        inner
            .as_ref()
            .map_or(SinkMpiSupport::RankLocal, |sink| sink.mpi_support())
    }
}

fn py_generated_sink_arg(sink: &Bound<'_, PyAny>) -> PyResult<AnyPySink> {
    if let Ok(dataset_sink) = sink.extract::<PyDatasetSink>() {
        return Ok(AnyPySink {
            inner: PyGeneratedSink::new(DatasetPySink {
                sink: dataset_sink.0,
            })
            .inner,
        });
    }
    if let Ok(native_sink) = sink.extract::<PyRef<'_, PyGeneratedSink>>() {
        return Ok(AnyPySink {
            inner: Rc::clone(&native_sink.inner),
        });
    }
    if let Ok(method) = sink.getattr("__laddu_sink__") {
        let native_sink = method.call0()?.extract::<PyRef<'_, PyGeneratedSink>>()?;
        return Ok(AnyPySink {
            inner: Rc::clone(&native_sink.inner),
        });
    }
    Err(PyTypeError::new_err(
        "expected sink to be a DatasetSink, GeneratedSink, or object with __laddu_sink__()",
    ))
}

#[pyclass(name = "EnvelopeStats", module = "laddu", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEnvelopeStats(pub EnvelopeStats);

#[pymethods]
impl PyEnvelopeStats {
    #[getter]
    fn configured_max(&self) -> Option<f64> {
        self.0.configured_max
    }

    #[getter]
    fn observed_max(&self) -> Option<f64> {
        self.0.observed_max
    }

    #[getter]
    fn violations(&self) -> u64 {
        self.0.violations
    }

    #[getter]
    fn largest_violation_ratio(&self) -> Option<f64> {
        self.0.largest_violation_ratio
    }

    #[getter]
    fn updates(&self) -> u64 {
        self.0.updates
    }

    #[getter]
    fn final_max(&self) -> Option<f64> {
        self.0.final_max
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
        self.0.envelope()
    }

    #[getter]
    fn envelope_violations(&self) -> u64 {
        self.0.envelope_violations()
    }

    #[getter]
    fn envelope_stats(&self) -> Option<PyEnvelopeStats> {
        self.0.envelope_stats.clone().map(PyEnvelopeStats)
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
pub struct PyGenerationResult(pub GenerationResult<Py<PyAny>>);

#[pymethods]
impl PyGenerationResult {
    #[getter]
    fn output(&self) -> Py<PyAny> {
        Python::attach(|py| self.0.output.clone_ref(py))
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
        sink: &Bound<'_, PyAny>,
        mode: Option<&PyGenerationMode>,
        options: Option<&PyGenerationOptions>,
    ) -> PyResult<PyGenerationResult> {
        let mode = mode.map(|mode| mode.0.clone()).unwrap_or_default();
        let options = options.map(|options| options.0.clone()).unwrap_or_default();
        let sink = py_generated_sink_arg(sink)?;
        Ok(PyGenerationResult(self.generator.generate(
            target_events,
            sink,
            mode,
            options,
        )?))
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
