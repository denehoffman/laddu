use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use laddu_fit::{
    AnalysisSnapshot, BoundFitState, FitArtifact, FitEnsembleArtifact, FitError, FitOutcome,
    FitProblem, MinimizationResult,
    ganesh::{
        NalgebraProvider, Vector,
        algorithms::{
            gradient::{
                Adam, AdamConfig, AdamEMATerminator, ConjugateGradient,
                ConjugateGradientGTerminator, GradientStatus, LBFGSB, LBFGSBConfig,
                LBFGSBFTerminator, LBFGSBGTerminator, LBFGSBInfNormGTerminator, TrustRegion,
                TrustRegionGTerminator,
            },
            gradient_free::{
                CMAES, CMAESConditionCovTerminator, CMAESEqualFunValuesTerminator,
                CMAESNoEffectAxisTerminator, CMAESNoEffectCoordTerminator, CMAESSigmaTerminator,
                CMAESStagnationTerminator, CMAESTolFunTerminator, CMAESTolXTerminator,
                CMAESTolXUpTerminator, DifferentialEvolution, GradientFreeStatus, NelderMead,
                NelderMeadFTerminator, NelderMeadXTerminator, SimulatedAnnealing,
                SimulatedAnnealingTerminator,
            },
            mcmc::{AIES, AIESConfig, AutocorrelationTerminator, ESS, ESSConfig, EnsembleStatus},
            particles::PSO,
        },
        core::{Callbacks, DebugObserver, MaxSteps, ProgressObserver},
        python::{
            PyAIESConfig, PyAIESInit, PyAdamConfig, PyAdamEMATerminator,
            PyAutocorrelationTerminator, PyCMAESConditionCovTerminator, PyCMAESConfig,
            PyCMAESEqualFunValuesTerminator, PyCMAESNoEffectAxisTerminator,
            PyCMAESNoEffectCoordTerminator, PyCMAESSigmaTerminator, PyCMAESStagnationTerminator,
            PyCMAESTolFunTerminator, PyCMAESTolXTerminator, PyCMAESTolXUpTerminator,
            PyConjugateGradientConfig, PyConjugateGradientGTerminator, PyDebugObserver,
            PyDifferentialEvolutionConfig, PyESSConfig, PyESSInit, PyLBFGSBConfig,
            PyLBFGSBFTerminator, PyLBFGSBGTerminator, PyLBFGSBInfNormGTerminator, PyMCMCSummary,
            PyMaxSteps, PyMinimizationSummary, PyNelderMeadConfig, PyNelderMeadFTerminator,
            PyNelderMeadXTerminator, PyPSOConfig, PyProgressObserver, PySimulatedAnnealingConfig,
            PySimulatedAnnealingTerminator, PyTrustRegionConfig, PyTrustRegionGTerminator,
            PyVectorInit, PythonCallbackBundle, process_with_python_callbacks,
        },
        traits::{Algorithm, CostFunction, Gradient, LogDensity, SupportsParameterNames},
    },
};
use laddu_likelihood::{
    BootstrapFitError, Ensemble, Likelihood, LikelihoodEvaluation, Objective, StochasticObjective,
};
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyAny, PyDict},
};

use super::{
    cross_section::{PyCrossSection, PyEnsemble, PyLuminosity, PyYield},
    data::PyDataset,
    error::to_py_err,
    likelihood::{PyLikelihood, free_values},
};

#[pyclass(name = "FitResult", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Stable result of deterministic minimization owned by `laddu`.
pub struct PyFitResult {
    inner: MinimizationResult,
}

impl PyFitResult {
    fn from_summary(
        summary: laddu_fit::ganesh::core::MinimizationSummary,
        parameter_names: &[String],
        fingerprint: String,
        strong_compatibility: bool,
    ) -> Self {
        Self {
            inner: MinimizationResult::from_ganesh(summary, parameter_names)
                .with_likelihood_fingerprint(fingerprint)
                .with_strong_compatibility(strong_compatibility),
        }
    }
}

#[pymethods]
impl PyFitResult {
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().to_vec()
    }
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.values().to_vec())
    }
    #[getter]
    fn named_parameters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let output = PyDict::new(py);
        for (name, value) in self.inner.parameter_names().iter().zip(self.inner.values()) {
            output.set_item(name, value)?;
        }
        Ok(output)
    }
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective()
    }
    #[getter]
    fn outcome(&self) -> &'static str {
        match self.inner.outcome() {
            FitOutcome::Converged => "converged",
            FitOutcome::NotConverged => "not_converged",
        }
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged()
    }
    #[getter]
    fn terminal_message(&self) -> &str {
        self.inner.terminal_message()
    }
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        Ok(self
            .inner
            .covariance()
            .map(|matrix| PyArray2::from_vec2(py, matrix))
            .transpose()?)
    }
    #[getter]
    fn standard_errors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .standard_errors()
            .map(|values| PyArray1::from_vec(py, values.to_vec()))
    }
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let diagnostics = self.inner.diagnostics();
        let output = PyDict::new(py);
        output.set_item("function_evaluations", diagnostics.function_evaluations())?;
        output.set_item("gradient_evaluations", diagnostics.gradient_evaluations())?;
        output.set_item("hessian_evaluations", diagnostics.hessian_evaluations())?;
        Ok(output)
    }
    #[getter]
    fn raw_ganesh_summary(&self) -> PyMinimizationSummary {
        self.inner.raw_ganesh_summary().clone().into()
    }
    fn artifact(&self) -> PyResult<PyFitArtifact> {
        self.inner
            .artifact()
            .map(|inner| PyFitArtifact { inner })
            .map_err(to_py_err)
    }
}

#[pyclass(name = "FitArtifact", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Dataset-free, versioned fit artifact.
pub struct PyFitArtifact {
    inner: FitArtifact,
}
#[pymethods]
impl PyFitArtifact {
    #[staticmethod]
    fn load(path: std::path::PathBuf) -> PyResult<Self> {
        FitArtifact::load(path)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }
    #[pyo3(signature = (path, *, overwrite=false))]
    fn save(&self, path: std::path::PathBuf, overwrite: bool) -> PyResult<()> {
        self.inner.save(path, overwrite).map_err(to_py_err)
    }
    fn with_ensemble(&self, ensemble: &PyEnsemble) -> PyResult<Self> {
        self.inner
            .clone()
            .with_ensemble(&ensemble.inner)
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }
    fn with_failed_replicas(&self, failures: Vec<(usize, Option<u64>, String)>) -> PyResult<Self> {
        self.inner
            .clone()
            .with_failed_replicas(
                failures
                    .into_iter()
                    .map(|(index, seed, message)| laddu_fit::FailedReplica {
                        index,
                        seed,
                        message,
                    })
                    .collect(),
            )
            .map(|inner| Self { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn ensemble(&self) -> Option<PyFitEnsembleArtifact> {
        self.inner
            .ensemble()
            .cloned()
            .map(|inner| PyFitEnsembleArtifact { inner })
    }
    #[pyo3(signature = (likelihood, term_name, *, generated_mc))]
    /// Bind explicitly, then reconstruct a yield context without optimization.
    fn yield_context(
        &self,
        likelihood: &PyLikelihood,
        term_name: &str,
        generated_mc: &PyDataset,
    ) -> PyResult<PyYield> {
        let bound = self.inner.bind(&likelihood.inner).map_err(to_py_err)?;
        let ensemble = self
            .inner
            .ensemble()
            .map(|ensemble| ensemble.to_ensemble_for(&likelihood.inner))
            .transpose()
            .map_err(to_py_err)?;
        laddu_likelihood::Yield::with_ensemble(
            std::sync::Arc::clone(&likelihood.inner),
            term_name,
            generated_mc.inner.clone(),
            bound.values().to_vec(),
            ensemble,
        )
        .map(Into::into)
        .map_err(to_py_err)
    }
    #[pyo3(signature = (likelihood, term_name, *, generated_mc, luminosity))]
    /// Bind the stored fit values and evaluate the fitted generated-MC cross section.
    fn cross_section(
        &self,
        likelihood: &PyLikelihood,
        term_name: &str,
        generated_mc: &PyDataset,
        luminosity: &PyLuminosity,
    ) -> PyResult<PyCrossSection> {
        let bound = self.inner.bind(&likelihood.inner).map_err(to_py_err)?;
        let ensemble = self
            .inner
            .ensemble()
            .map(|ensemble| ensemble.to_ensemble_for(&likelihood.inner))
            .transpose()
            .map_err(to_py_err)?;
        likelihood
            .inner
            .cross_section(
                term_name,
                generated_mc.inner.clone(),
                luminosity.inner.clone(),
                bound.values().to_vec(),
                ensemble,
            )
            .map(|inner| PyCrossSection { inner })
            .map_err(to_py_err)
    }
    #[getter]
    fn artifact_kind(&self) -> &str {
        self.inner.artifact_kind()
    }
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version()
    }
    #[getter]
    fn laddu_version(&self) -> &str {
        self.inner.laddu_version()
    }
    #[getter]
    fn fingerprint_version(&self) -> u32 {
        self.inner.fingerprint_version()
    }
    #[getter]
    fn likelihood_fingerprint(&self) -> &str {
        self.inner.likelihood_fingerprint()
    }
    #[getter]
    fn strong_compatibility(&self) -> bool {
        self.inner.strong_compatibility()
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().to_vec()
    }
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.values().to_vec())
    }
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective()
    }
    #[getter]
    fn outcome(&self) -> &'static str {
        match self.inner.outcome() {
            FitOutcome::Converged => "converged",
            FitOutcome::NotConverged => "not_converged",
        }
    }
    #[getter]
    fn terminal_message(&self) -> &str {
        self.inner.terminal_message()
    }
    #[getter]
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let diagnostics = self.inner.diagnostics();
        let output = PyDict::new(py);
        output.set_item("function_evaluations", diagnostics.function_evaluations())?;
        output.set_item("gradient_evaluations", diagnostics.gradient_evaluations())?;
        output.set_item("hessian_evaluations", diagnostics.hessian_evaluations())?;
        Ok(output)
    }
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        Ok(self
            .inner
            .covariance()
            .map(|matrix| PyArray2::from_vec2(py, matrix))
            .transpose()?)
    }
    #[getter]
    fn standard_errors<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .standard_errors()
            .map(|values| PyArray1::from_vec(py, values.to_vec()))
    }
    fn bind(&self, likelihood: &PyLikelihood) -> PyResult<PyBoundFitState> {
        self.inner
            .bind(&likelihood.inner)
            .map(|inner| PyBoundFitState { inner })
            .map_err(to_py_err)
    }
    fn bind_with_parameter_map(
        &self,
        likelihood: &PyLikelihood,
        mapping: HashMap<String, String>,
    ) -> PyResult<PyBoundFitState> {
        self.inner
            .bind_with_parameter_map(&likelihood.inner, &mapping)
            .map(|inner| PyBoundFitState { inner })
            .map_err(to_py_err)
    }
}

#[pyclass(
    name = "FitEnsembleArtifact",
    module = "laddu",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
/// Dataset-free ordered ensemble contents and pairing identity.
pub struct PyFitEnsembleArtifact {
    inner: FitEnsembleArtifact,
}
#[pymethods]
impl PyFitEnsembleArtifact {
    #[getter]
    fn source_id(&self) -> u64 {
        self.inner.source_id()
    }
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().to_vec()
    }
    #[getter]
    fn draws(&self) -> Vec<Vec<f64>> {
        self.inner.draws().to_vec()
    }
    #[getter]
    fn draw_ids(&self) -> Vec<u64> {
        self.inner.draw_ids().to_vec()
    }
    #[getter]
    fn bootstrap_seed(&self) -> Option<u64> {
        self.inner.bootstrap_seed()
    }
    #[getter]
    fn requires_external_datasets(&self) -> bool {
        self.inner.requires_external_datasets()
    }
    #[getter]
    fn failures(&self) -> Vec<(usize, Option<u64>, String)> {
        self.inner
            .failures()
            .iter()
            .map(|failure| (failure.index, failure.seed, failure.message.clone()))
            .collect()
    }
    fn to_ensemble(&self) -> PyResult<PyEnsemble> {
        self.inner
            .to_parameter_ensemble()
            .map(|inner| PyEnsemble { inner })
            .map_err(to_py_err)
    }
}

#[pyclass(name = "BoundFitState", module = "laddu", frozen, skip_from_py_object)]
#[derive(Clone)]
/// Fit state validated and reordered for a reconstructed likelihood.
pub struct PyBoundFitState {
    inner: BoundFitState,
}

#[pyclass(name = "AnalysisSnapshot", module = "laddu")]
#[derive(Default)]
/// Passive object graph for shared fit-artifact references.
pub struct PyAnalysisSnapshot {
    fits: HashMap<String, Py<PyFitArtifact>>,
}
#[pymethods]
impl PyAnalysisSnapshot {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    fn add_fit(&mut self, path: String, artifact: Py<PyFitArtifact>) -> PyResult<()> {
        if path.is_empty() || path.contains("..") {
            return Err(to_py_err(format!("invalid snapshot object path `{path}`")));
        }
        if self.fits.contains_key(&path) {
            return Err(to_py_err(format!(
                "duplicate snapshot object path `{path}`"
            )));
        }
        self.fits.insert(path, artifact);
        Ok(())
    }
    fn alias_fit(&mut self, py: Python<'_>, path: String, existing: &str) -> PyResult<()> {
        let artifact = self
            .fits
            .get(existing)
            .map(|artifact| artifact.clone_ref(py))
            .ok_or_else(|| to_py_err(format!("snapshot fit path `{existing}` is missing")))?;
        self.add_fit(path, artifact)
    }
    fn fit(&self, py: Python<'_>, path: &str) -> PyResult<Py<PyFitArtifact>> {
        self.fits
            .get(path)
            .map(|artifact| artifact.clone_ref(py))
            .ok_or_else(|| to_py_err(format!("snapshot fit path `{path}` is missing")))
    }
    #[pyo3(signature = (path, *, overwrite=false))]
    fn save(&self, py: Python<'_>, path: std::path::PathBuf, overwrite: bool) -> PyResult<()> {
        let mut snapshot = AnalysisSnapshot::new();
        let mut shared = HashMap::<usize, Arc<FitArtifact>>::new();
        for (object_path, artifact) in &self.fits {
            let pointer = artifact.as_ptr() as usize;
            let inner = shared
                .entry(pointer)
                .or_insert_with(|| Arc::new(artifact.borrow(py).inner.clone()))
                .clone();
            snapshot
                .insert_fit(object_path.clone(), inner)
                .map_err(to_py_err)?;
        }
        snapshot.save(path, overwrite).map_err(to_py_err)
    }
    #[staticmethod]
    fn load(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let snapshot = AnalysisSnapshot::load(path).map_err(to_py_err)?;
        let mut fits = HashMap::new();
        let mut shared = HashMap::<usize, Py<PyFitArtifact>>::new();
        for object_path in snapshot.fit_paths() {
            let artifact = snapshot.fit(object_path).expect("path came from snapshot");
            let pointer = Arc::as_ptr(artifact) as usize;
            let object = if let Some(object) = shared.get(&pointer) {
                object.clone_ref(py)
            } else {
                let object = Py::new(
                    py,
                    PyFitArtifact {
                        inner: artifact.as_ref().clone(),
                    },
                )?;
                shared.insert(pointer, object.clone_ref(py));
                object
            };
            fits.insert(object_path.to_owned(), object);
        }
        Ok(Self { fits })
    }
}
#[pymethods]
impl PyBoundFitState {
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameter_names().to_vec()
    }
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.values().to_vec())
    }
    #[getter]
    fn objective(&self) -> f64 {
        self.inner.objective()
    }
    #[getter]
    fn outcome(&self) -> &'static str {
        match self.inner.outcome() {
            FitOutcome::Converged => "converged",
            FitOutcome::NotConverged => "not_converged",
        }
    }
    #[getter]
    fn migration(&self) -> Vec<(String, String)> {
        self.inner.migration().to_vec()
    }
}

#[derive(Clone)]
struct OwnedProblem {
    objective: Arc<Likelihood>,
}

impl OwnedProblem {
    fn new(objective: Arc<Likelihood>) -> Self {
        Self { objective }
    }
    fn metadata(&self) -> FitProblem<'_, Likelihood, f64> {
        FitProblem::new(&self.objective)
    }
    fn external(x: &Vector) -> Vec<f64> {
        (0..x.len()).map(|index| x.get(index)).collect()
    }
}

impl CostFunction<f64, NalgebraProvider, (), FitError> for OwnedProblem {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, FitError> {
        Ok(self.objective.value(&Self::external(x))?)
    }
}

impl Gradient<f64, NalgebraProvider, (), FitError> for OwnedProblem {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, FitError> {
        Ok(Vector::from_vec(
            self.objective
                .value_gradient(&Self::external(x))?
                .gradient()
                .to_vec(),
        ))
    }

    fn evaluate_with_gradient(&self, x: &Vector, _: &()) -> Result<(f64, Vector), FitError> {
        let evaluation = self.objective.value_gradient(&Self::external(x))?;
        Ok((
            evaluation.value(),
            Vector::from_vec(evaluation.gradient().to_vec()),
        ))
    }
}

impl LogDensity<f64, NalgebraProvider, (), FitError> for OwnedProblem {
    fn log_density(&self, x: &Vector, _: &()) -> Result<f64, FitError> {
        let external = Self::external(x);
        if self
            .objective
            .parameter_layout()
            .validate_free_values(&external)
            .is_err()
        {
            return Ok(f64::NEG_INFINITY);
        }
        Ok(-self.objective.value(&external)?)
    }
}

struct OwnedStochasticProblem {
    objective: Arc<Likelihood>,
    fraction: f64,
    next_seed: AtomicU64,
    cache: parking_lot::Mutex<Option<(Vec<f64>, LikelihoodEvaluation)>>,
}

impl OwnedStochasticProblem {
    fn evaluation(&self, x: &Vector, consume: bool) -> Result<LikelihoodEvaluation, FitError> {
        let values = OwnedProblem::external(x);
        let mut cache = self.cache.lock();
        if let Some((point, evaluation)) = cache.as_ref()
            && point == &values
        {
            let evaluation = evaluation.clone();
            if consume {
                *cache = None;
            }
            return Ok(evaluation);
        }
        let seed = self.next_seed.fetch_add(1, Ordering::Relaxed);
        let evaluation = self
            .objective
            .stochastic_value_gradient(&values, self.fraction, seed)?;
        *cache = if consume {
            None
        } else {
            Some((values, evaluation.clone()))
        };
        Ok(evaluation)
    }
}

impl CostFunction<f64, NalgebraProvider, (), FitError> for OwnedStochasticProblem {
    fn evaluate(&self, x: &Vector, _: &()) -> Result<f64, FitError> {
        Ok(self.evaluation(x, false)?.value())
    }
}

impl Gradient<f64, NalgebraProvider, (), FitError> for OwnedStochasticProblem {
    fn gradient(&self, x: &Vector, _: &()) -> Result<Vector, FitError> {
        Ok(Vector::from_vec(
            self.evaluation(x, true)?.gradient().to_vec(),
        ))
    }

    fn evaluate_with_gradient(&self, x: &Vector, _: &()) -> Result<(f64, Vector), FitError> {
        let evaluation = self.evaluation(x, true)?;
        Ok((
            evaluation.value(),
            Vector::from_vec(evaluation.gradient().to_vec()),
        ))
    }
}

fn initial_vector(likelihood: &Likelihood, initial: Option<&Bound<'_, PyAny>>) -> PyResult<Vector> {
    let values = match initial {
        None => likelihood.default_params(),
        Some(value) if value.extract::<PyRef<'_, PyVectorInit>>().is_ok() => value
            .extract::<PyRef<'_, PyVectorInit>>()?
            .to_rust()
            .to_vec(),
        Some(value) => free_values(likelihood, value)?,
    };
    Ok(Vector::from_vec(values))
}

// Convert callbacks against the selected algorithm so incompatible terminators
// fail before optimization, while Python callbacks retain exception propagation.
macro_rules! algorithm_callbacks {
    (@typed $algorithm:ident, $status:ident, $py:expr, $defaults:expr,
        $terminators:expr, $observers:expr; $($python:ident => $rust:ident),* $(,)?) => {{
        let callbacks = if $defaults {
            $algorithm::default_callbacks()
        } else {
            Callbacks::empty()
        };
        let mut bundle = callback_bundle::<$algorithm, _, $status, _>(
            $py, callbacks, Vec::new(), $observers
        )?;
        for terminator in $terminators {
            $(
                if let Ok(value) = terminator.bind($py).extract::<PyRef<'_, $python>>() {
                    bundle = bundle.with_terminator(
                        $rust::try_from(value.clone()).map_err(to_py_err)?
                    );
                    continue;
                }
            )*
            bundle = common_terminator($py, bundle, terminator)?;
        }
        Ok::<_, PyErr>(bundle)
    }};
    (Adam, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed Adam, GradientStatus, $py, $defaults, $terminators, $observers;
            PyAdamEMATerminator => AdamEMATerminator)
    };
    (ConjugateGradient, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed ConjugateGradient, GradientStatus, $py, $defaults, $terminators, $observers;
            PyConjugateGradientGTerminator => ConjugateGradientGTerminator)
    };
    (LBFGSB, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed LBFGSB, GradientStatus, $py, $defaults, $terminators, $observers;
            PyLBFGSBFTerminator => LBFGSBFTerminator, PyLBFGSBGTerminator => LBFGSBGTerminator, PyLBFGSBInfNormGTerminator => LBFGSBInfNormGTerminator)
    };
    (TrustRegion, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed TrustRegion, GradientStatus, $py, $defaults, $terminators, $observers;
            PyTrustRegionGTerminator => TrustRegionGTerminator)
    };
    (NelderMead, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed NelderMead, GradientFreeStatus, $py, $defaults, $terminators, $observers;
            PyNelderMeadFTerminator => NelderMeadFTerminator, PyNelderMeadXTerminator => NelderMeadXTerminator)
    };
    (CMAES, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed CMAES, GradientFreeStatus, $py, $defaults, $terminators, $observers;
            PyCMAESSigmaTerminator => CMAESSigmaTerminator, PyCMAESNoEffectAxisTerminator => CMAESNoEffectAxisTerminator, PyCMAESNoEffectCoordTerminator => CMAESNoEffectCoordTerminator, PyCMAESConditionCovTerminator => CMAESConditionCovTerminator, PyCMAESEqualFunValuesTerminator => CMAESEqualFunValuesTerminator, PyCMAESStagnationTerminator => CMAESStagnationTerminator, PyCMAESTolXUpTerminator => CMAESTolXUpTerminator, PyCMAESTolFunTerminator => CMAESTolFunTerminator, PyCMAESTolXTerminator => CMAESTolXTerminator)
    };
    (DifferentialEvolution, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed DifferentialEvolution, GradientFreeStatus, $py, $defaults, $terminators, $observers;
            )
    };
    (SimulatedAnnealing, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed SimulatedAnnealing, GradientFreeStatus, $py, $defaults, $terminators, $observers;
            PySimulatedAnnealingTerminator => SimulatedAnnealingTerminator)
    };
    (PSO, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed PSO, GradientFreeStatus, $py, $defaults, $terminators, $observers;
            )
    };
    (AIES, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed AIES, EnsembleStatus, $py, $defaults, $terminators, $observers;
            PyAutocorrelationTerminator => AutocorrelationTerminator)
    };
    (ESS, $py:expr, $defaults:expr, $terminators:expr, $observers:expr) => {
        algorithm_callbacks!(@typed ESS, EnsembleStatus, $py, $defaults, $terminators, $observers;
            PyAutocorrelationTerminator => AutocorrelationTerminator)
    };
}

fn common_terminator<A, P, S, C>(
    py: Python<'_>,
    mut bundle: PythonCallbackBundle<A, P, S, (), FitError, C>,
    terminator: Py<PyAny>,
) -> PyResult<PythonCallbackBundle<A, P, S, (), FitError, C>>
where
    A: Algorithm<P, S, (), FitError, Config = C> + 'static,
    P: 'static,
    S: laddu_fit::ganesh::traits::Status + 'static,
    C: 'static,
{
    if let Ok(value) = terminator.bind(py).extract::<PyRef<'_, PyMaxSteps>>() {
        bundle = bundle.with_terminator(MaxSteps::from(*value));
    } else if terminator.bind(py).is_callable() {
        bundle = bundle.with_python_terminator(terminator);
    } else {
        return Err(PyTypeError::new_err(
            "terminators must match the selected ganesh algorithm, be ganesh.MaxSteps, or be Python callables",
        ));
    }
    Ok(bundle)
}

fn callback_bundle<A, P, S, C>(
    py: Python<'_>,
    callbacks: Callbacks<A, P, S, (), FitError, C>,
    terminators: Vec<Py<PyAny>>,
    observers: Vec<Py<PyAny>>,
) -> PyResult<PythonCallbackBundle<A, P, S, (), FitError, C>>
where
    A: Algorithm<P, S, (), FitError, Config = C> + 'static,
    P: 'static,
    S: laddu_fit::ganesh::traits::Status
        + laddu_fit::ganesh::traits::ProgressStatus
        + std::fmt::Debug
        + 'static,
    C: 'static,
{
    let mut bundle = PythonCallbackBundle::new(callbacks);
    for terminator in terminators {
        bundle = common_terminator(py, bundle, terminator)?;
    }
    for observer in observers {
        if let Ok(value) = observer.bind(py).extract::<PyRef<'_, PyProgressObserver>>() {
            bundle = bundle.with_observer(ProgressObserver::from(*value));
        } else if let Ok(value) = observer.bind(py).extract::<PyRef<'_, PyDebugObserver>>() {
            bundle = bundle.with_observer(DebugObserver::from(*value));
        } else if observer.bind(py).is_callable() {
            bundle = bundle.with_python_observer(observer);
        } else {
            return Err(PyTypeError::new_err(
                "observers must be ganesh observer objects or Python callables",
            ));
        }
    }
    Ok(bundle)
}

#[pymethods]
impl PyLikelihood {
    #[pyo3(signature = (
        center: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float]",
        n_walkers,
        *,
        scale=1.0e-3,
        seed=0
    ) -> "Sequence[Sequence[float]]")]
    /// Generate walker positions in a small cloud around a fitted point.
    ///
    /// The returned NumPy array has shape ``(n_walkers, n_parameters)``.
    /// Per-parameter scales declared on model parameters are used when
    /// available; otherwise the characteristic scale is
    /// ``max(abs(center), 1)``. Periodic coordinates are wrapped and bounded
    /// proposals are resampled.
    ///
    /// Parameters
    /// ----------
    /// center : sequence, numpy.ndarray, or dict[str, float]
    ///     Center in free-parameter coordinates.
    /// n_walkers : int
    ///     Number of walkers. AIES requires at least two.
    /// scale : float, default=1e-3
    ///     Fractional half-width of the uniform cloud.
    /// seed : int, default=0
    ///     Random seed.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Initial walker matrix suitable for ``ganesh.AIESInit``.
    fn walker_positions<'py>(
        &self,
        py: Python<'py>,
        center: &Bound<'_, PyAny>,
        n_walkers: usize,
        scale: f64,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if n_walkers < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n_walkers must be at least 2",
            ));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "scale must be finite and positive",
            ));
        }
        let center = free_values(&self.inner, center)?;
        self.inner
            .params()
            .validate_free_values(&center)
            .map_err(to_py_err)?;
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut rows = Vec::with_capacity(n_walkers);
        for _ in 0..n_walkers {
            let mut row = center.clone();
            for (index, id) in self.inner.params().free_params().iter().enumerate() {
                let parameter = self.inner.params().spec(*id).map_err(to_py_err)?;
                let width = scale
                    * parameter
                        .scale()
                        .unwrap_or_else(|| center[index].abs().max(1.0));
                let mut accepted = None;
                for _ in 0..128 {
                    let candidate = center[index] + width * (2.0 * rng.f64() - 1.0);
                    if parameter.bounds_spec().contains(candidate) {
                        accepted = Some(candidate);
                        break;
                    }
                }
                row[index] = accepted.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "could not generate a bounded walker coordinate for `{}`; reduce scale or move the center away from the boundary",
                        parameter.name()
                    ))
                })?;
            }
            row = self
                .inner
                .params()
                .wrap_periodic_free_values(&row)
                .map_err(to_py_err)?;
            rows.push(row);
        }
        Ok(PyArray2::from_vec2(py, &rows)?)
    }

    #[pyo3(signature = (
        initial: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        *,
        config: "object | None" = None,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new(),
        seed=0,
        default_callbacks=true,
        use_parameter_metadata=true
    ))]
    /// Minimize this likelihood with any ganesh minimizer.
    ///
    /// Parameters
    /// ----------
    /// initial : sequence, numpy.ndarray, or dict[str, float], optional
    ///     Initial free-parameter values. A mapping may specify only the names
    ///     to override; remaining parameters use their defaults.
    /// config : object, optional
    ///     Any ganesh minimizer configuration: Adam, ConjugateGradient,
    ///     LBFGSB, TrustRegion, NelderMead, CMAES, DifferentialEvolution,
    ///     SimulatedAnnealing, or PSO. ``None`` uses L-BFGS-B with analytic
    ///     gradients and the bounds declared by model parameters.
    /// terminators : sequence, optional
    ///     ganesh termination callbacks, such as ``ganesh.MaxSteps``.
    /// observers : sequence, optional
    ///     ganesh observers or Python callables invoked during minimization.
    /// seed : int, default=0
    ///     Seed for randomized minimizers.
    /// default_callbacks : bool, default=True
    ///     Include ganesh's default termination criteria. Set False to use
    ///     only the supplied callbacks; provide a stopping condition.
    /// use_parameter_metadata : bool, default=True
    ///     Install names, transforms, and native bounds from the likelihood.
    ///     Set False to pass these configuration fields through unchanged.
    ///     Algorithm settings, including line searches, are always preserved.
    ///
    /// Returns
    /// -------
    /// FitResult
    ///     Stable parameters, objective, terminal outcome, inference metadata,
    ///     and explicit access to the complete Ganesh summary.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the configuration, initial values, or callbacks are unsupported.
    /// LadduError
    ///     If parameter transformation or likelihood evaluation fails.
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &self,
        py: Python<'_>,
        initial: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
        seed: u64,
        default_callbacks: bool,
        use_parameter_metadata: bool,
    ) -> PyResult<PyFitResult> {
        let initial = initial_vector(&self.inner, initial)?;
        let problem = OwnedProblem::new(Arc::clone(&self.inner));
        let metadata = problem.metadata();
        macro_rules! run {
            ($algorithm:ident, $config:expr, $optimizer:expr, $native:ident) => {{
                let config = $config;
                let config = if use_parameter_metadata {
                    run!(@metadata config, $native)
                } else {
                    config
                };
                let callbacks = algorithm_callbacks!(
                    $algorithm, py, default_callbacks, terminators, observers
                )?;
                let summary = process_with_python_callbacks(
                    &mut $optimizer, &problem, &(), initial, config, callbacks, to_py_err,
                )?;
                return Ok(PyFitResult::from_summary(
                    summary, &metadata.parameter_names(),
                    self.inner.artifact_fingerprint_v1(),
                    self.inner.artifact_compatibility_is_strong(),
                ));
            }};
            (@metadata $config:ident, native) => {
                $config.with_parameter_names(metadata.parameter_names())
                    .with_transform(metadata.native_transform().map_err(to_py_err)?)
                    .map_err(to_py_err)?
                    .with_bounds(metadata.native_bounds()).map_err(to_py_err)?
            };
            (@metadata $config:ident, smooth) => {
                $config.with_parameter_names(metadata.parameter_names())
                    .with_transform(metadata.minimizer_transform().map_err(to_py_err)?)
            };
        }
        if let Some(config) = config {
            if let Ok(config) = config.extract::<PyRef<'_, PyAdamConfig>>() {
                run!(
                    Adam,
                    config.to_rust().map_err(to_py_err)?,
                    Adam::default(),
                    smooth
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PyConjugateGradientConfig>>() {
                run!(
                    ConjugateGradient,
                    config.to_rust().map_err(to_py_err)?,
                    ConjugateGradient::default(),
                    smooth
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PyLBFGSBConfig>>() {
                run!(
                    LBFGSB,
                    config.to_rust().map_err(to_py_err)?,
                    LBFGSB::default(),
                    native
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PyTrustRegionConfig>>() {
                run!(
                    TrustRegion,
                    config.to_rust().map_err(to_py_err)?,
                    TrustRegion::default(),
                    smooth
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PyNelderMeadConfig>>() {
                run!(
                    NelderMead,
                    config.to_rust().map_err(to_py_err)?,
                    NelderMead::default(),
                    smooth
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PyCMAESConfig>>() {
                run!(
                    CMAES,
                    config.to_rust().map_err(to_py_err)?,
                    CMAES::new(Some(seed)),
                    smooth
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PyDifferentialEvolutionConfig>>() {
                run!(
                    DifferentialEvolution,
                    config.to_rust().map_err(to_py_err)?,
                    DifferentialEvolution::new(Some(seed)),
                    smooth
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PySimulatedAnnealingConfig>>() {
                run!(
                    SimulatedAnnealing,
                    config.to_rust().map_err(to_py_err)?,
                    SimulatedAnnealing::new(Some(seed)),
                    smooth
                );
            }
            if let Ok(config) = config.extract::<PyRef<'_, PyPSOConfig>>() {
                run!(
                    PSO,
                    config.to_rust().map_err(to_py_err)?,
                    PSO::new(Some(seed)),
                    smooth
                );
            }
            return Err(PyTypeError::new_err(
                "fit config must be a ganesh minimizer configuration or None",
            ));
        }
        run!(LBFGSB, LBFGSBConfig::default(), LBFGSB::default(), native);
    }

    #[pyo3(signature = (
        restarts,
        initial: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        *,
        config: "object | None" = None,
        seed=0,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new(),
        default_callbacks=true,
        use_parameter_metadata=true
    ))]
    /// Run a deterministic ensemble of independent fit restarts.
    ///
    /// The first restart uses ``initial`` when supplied. Remaining starts are
    /// reproducibly sampled from parameter initialization ranges. Fits execute
    /// serially so Python callbacks and per-fit thread pools are not oversubscribed.
    /// Configurations, observers, and the default_callbacks and
    /// use_parameter_metadata controls have the same meaning as in ``fit``.
    #[allow(clippy::too_many_arguments)]
    fn fit_restarts(
        &self,
        py: Python<'_>,
        restarts: usize,
        initial: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
        default_callbacks: bool,
        use_parameter_metadata: bool,
    ) -> PyResult<Vec<PyFitResult>> {
        if restarts == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "restarts must be positive",
            ));
        }
        let mut summaries = Vec::with_capacity(restarts);
        for index in 0..restarts {
            let callbacks = terminators
                .iter()
                .map(|callback| callback.clone_ref(py))
                .collect();
            let summary = if index == 0 && initial.is_some() {
                self.fit(
                    py,
                    initial,
                    config,
                    callbacks,
                    observers.iter().map(|o| o.clone_ref(py)).collect(),
                    seed,
                    default_callbacks,
                    use_parameter_metadata,
                )?
            } else {
                let start = self.inner.sample_initial(
                    seed.wrapping_add((index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
                let start = PyArray1::from_vec(py, start).into_any();
                self.fit(
                    py,
                    Some(&start),
                    config,
                    callbacks,
                    observers.iter().map(|o| o.clone_ref(py)).collect(),
                    seed,
                    default_callbacks,
                    use_parameter_metadata,
                )?
            };
            summaries.push(summary);
        }
        Ok(summaries)
    }

    #[pyo3(signature = (
        initial: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        *,
        config=None,
        fraction=0.1,
        seed=0,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new(),
        default_callbacks=true,
        use_parameter_metadata=true
    ))]
    #[allow(clippy::too_many_arguments)]
    /// Minimize this likelihood with stochastic Adam updates.
    ///
    /// Parameters
    /// ----------
    /// initial : sequence, numpy.ndarray, or dict[str, float], optional
    ///     Initial free-parameter values.
    /// config : ganesh.AdamConfig, optional
    ///     Adam optimizer configuration. ``None`` uses the default
    ///     configuration.
    /// fraction : float, default=0.1
    ///     Fraction of events sampled for each stochastic evaluation. Must lie
    ///     in ``(0, 1]``.
    /// seed : int, default=0
    ///     Seed for reproducible event subsampling.
    /// terminators, observers : sequence, optional
    ///     ganesh callbacks applied during optimization, including AdamEMATerminator.
    /// default_callbacks : bool, default=True
    ///     Include ganesh's default Adam EMA termination criterion.
    /// use_parameter_metadata : bool, default=True
    ///     Install likelihood names and transforms. False preserves the config.
    ///
    /// Values and gradients at an update use the same event batch. Each
    /// gradient consumes the cached pair, so an unchanged point can draw a new
    /// batch on the next update. Equal seeds reproduce the batch sequence.
    ///
    /// Returns
    /// -------
    /// ganesh.MinimizationSummary
    ///     Final optimizer state and convergence metadata.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `fraction` is outside ``(0, 1]``.
    /// LadduError
    ///     If preparation or an objective evaluation fails.
    fn fit_stochastic(
        &self,
        py: Python<'_>,
        initial: Option<&Bound<'_, PyAny>>,
        config: Option<&PyAdamConfig>,
        fraction: f64,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
        default_callbacks: bool,
        use_parameter_metadata: bool,
    ) -> PyResult<PyMinimizationSummary> {
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fraction must be in (0, 1]",
            ));
        }
        let initial = initial_vector(&self.inner, initial)?;
        let metadata = FitProblem::<_, f64>::new(&*self.inner);
        let config = match config {
            Some(config) => config.to_rust().map_err(to_py_err)?,
            None => AdamConfig::default(),
        };
        let config = if use_parameter_metadata {
            config
                .with_parameter_names(metadata.parameter_names())
                .with_transform(metadata.minimizer_transform().map_err(to_py_err)?)
        } else {
            config
        };
        let problem = OwnedStochasticProblem {
            objective: Arc::clone(&self.inner),
            fraction,
            next_seed: AtomicU64::new(seed),
            cache: parking_lot::Mutex::new(None),
        };
        let callbacks = algorithm_callbacks!(Adam, py, default_callbacks, terminators, observers)?;
        let summary = process_with_python_callbacks(
            &mut Adam::default(),
            &problem,
            &(),
            initial,
            config,
            callbacks,
            to_py_err,
        )?;
        Ok(summary.into())
    }

    #[pyo3(signature = (
        samples,
        initial: "Sequence[float] | numpy.typing.NDArray[numpy.float32 | numpy.float64] | dict[str, float] | None" = None,
        *,
        config: "object | None" = None,
        seed=0,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new(),
        default_callbacks=true,
        use_parameter_metadata=true
    ))]
    /// Poisson-bootstrap observed datasets, refit each replica, and retain the
    /// paired likelihood and parameter draws for cross-section propagation.
    /// Configurations, observers, and the default_callbacks and
    /// use_parameter_metadata controls are forwarded to each ``fit``.
    #[allow(clippy::too_many_arguments)]
    fn bootstrap_fit(
        &self,
        py: Python<'_>,
        samples: usize,
        initial: Option<&Bound<'_, PyAny>>,
        config: Option<&Bound<'_, PyAny>>,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
        default_callbacks: bool,
        use_parameter_metadata: bool,
    ) -> PyResult<PyEnsemble> {
        let inner = Ensemble::bootstrap_fit(&self.inner, samples, seed, |replica, _| {
            let replica_python = PyLikelihood {
                inner: Arc::clone(replica),
            };
            let callbacks = terminators
                .iter()
                .map(|callback| callback.clone_ref(py))
                .collect();
            let summary = replica_python.fit(
                py,
                initial,
                config,
                callbacks,
                observers.iter().map(|o| o.clone_ref(py)).collect(),
                seed,
                default_callbacks,
                use_parameter_metadata,
            )?;
            Ok(summary.inner.values().to_vec())
        })
        .map_err(|error| match error {
            BootstrapFitError::Likelihood(error) => to_py_err(error),
            BootstrapFitError::Fit { source, .. } => source,
        })?;
        Ok(PyEnsemble { inner })
    }

    #[pyo3(signature = (
        init,
        *,
        config: "object | None" = None,
        seed=0,
        terminators: "Sequence[object]" = Vec::new(),
        observers: "Sequence[object]" = Vec::new(),
        default_callbacks=true,
        use_parameter_metadata=true
    ))]
    /// Sample the likelihood with AIES or the ensemble slice sampler (ESS).
    ///
    /// Parameters
    /// ----------
    /// init : ganesh.AIESInit or ganesh.ESSInit
    ///     Initial walker ensemble; selects the sampler.
    /// config : ganesh.AIESConfig or ganesh.ESSConfig, optional
    ///     Matching sampler configuration. ``None`` uses its defaults.
    ///     Move mixtures, chain storage, and tuning settings pass through.
    /// seed : int, default=0
    ///     Random seed for proposal generation.
    /// terminators, observers : sequence, optional
    ///     ganesh callbacks, including AutocorrelationTerminator.
    /// default_callbacks : bool, default=True
    ///     Include ganesh's default sampler termination criteria.
    /// use_parameter_metadata : bool, default=True
    ///     Install likelihood names and linear scaling. False preserves the
    ///     supplied config, including transforms. Nonlinear transforms need
    ///     appropriate Jacobian handling to preserve a posterior density.
    ///
    /// Returns
    /// -------
    /// ganesh.MCMCSummary
    ///     Samples, acceptance diagnostics, and final ensemble state.
    ///
    /// Raises
    /// ------
    /// LadduError
    ///     If parameter transformation, initialization, or sampling fails.
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        init: &Bound<'_, PyAny>,
        config: Option<&Bound<'_, PyAny>>,
        seed: u64,
        terminators: Vec<Py<PyAny>>,
        observers: Vec<Py<PyAny>>,
        default_callbacks: bool,
        use_parameter_metadata: bool,
    ) -> PyResult<PyMCMCSummary> {
        let problem = OwnedProblem::new(Arc::clone(&self.inner));
        let metadata = problem.metadata();
        macro_rules! run_sampler {
            ($algorithm:ident, $pyinit:ident, $pyconfig:ident, $config:ident) => {{
                let init = init
                    .extract::<PyRef<'_, $pyinit>>()?
                    .to_rust()
                    .map_err(to_py_err)?;
                let config = match config {
                    Some(config) => config
                        .extract::<PyRef<'_, $pyconfig>>()?
                        .to_rust()
                        .map_err(to_py_err)?,
                    None => $config::default(),
                };
                let config = if use_parameter_metadata {
                    config
                        .with_parameter_names(metadata.parameter_names())
                        .with_transform(metadata.sampler_transform().map_err(to_py_err)?)
                } else {
                    config
                };
                let callbacks = algorithm_callbacks!(
                    $algorithm,
                    py,
                    default_callbacks,
                    terminators,
                    observers
                )?;
                let summary = process_with_python_callbacks(
                    &mut $algorithm::new(Some(seed)),
                    &problem,
                    &(),
                    init,
                    config,
                    callbacks,
                    to_py_err,
                )?;
                return Ok(summary.into());
            }};
        }
        if init.extract::<PyRef<'_, PyESSInit>>().is_ok() {
            run_sampler!(ESS, PyESSInit, PyESSConfig, ESSConfig);
        }
        run_sampler!(AIES, PyAIESInit, PyAIESConfig, AIESConfig);
    }
}
