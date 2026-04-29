use std::{collections::HashMap, sync::Arc};

use laddu_generation::{
    Distribution, EventGenerator, FinalStateParticle, GenComposite, GenFinalState, GenInitialState,
    GenReaction, InitialStateParticle, MandelstamTDistribution, Reconstruction,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};

use crate::{data::PyDataset, vectors::PyVec4};

/// A scalar distribution used by generated auxiliary columns.
#[pyclass(name = "Distribution", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyDistribution(pub Distribution);

#[pymethods]
impl PyDistribution {
    /// Construct a fixed scalar distribution.
    #[staticmethod]
    fn fixed(value: f64) -> Self {
        Self(Distribution::Fixed(value))
    }

    /// Construct a uniform scalar distribution.
    #[staticmethod]
    fn uniform(min: f64, max: f64) -> PyResult<Self> {
        if max <= min {
            return Err(PyValueError::new_err(
                "`max` must be greater than `min` for a uniform distribution",
            ));
        }
        Ok(Self(Distribution::Uniform { min, max }))
    }

    /// Construct a normal scalar distribution.
    #[staticmethod]
    fn normal(mu: f64, sigma: f64) -> PyResult<Self> {
        if sigma <= 0.0 {
            return Err(PyValueError::new_err(
                "`sigma` must be positive for a normal distribution",
            ));
        }
        Ok(Self(Distribution::Normal { mu, sigma }))
    }

    /// Construct an exponential scalar distribution.
    #[staticmethod]
    fn exponential(slope: f64) -> PyResult<Self> {
        if slope <= 0.0 {
            return Err(PyValueError::new_err(
                "`slope` must be positive for an exponential distribution",
            ));
        }
        Ok(Self(Distribution::Exponential { slope }))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A Mandelstam-t distribution for generated two-to-two reactions.
#[pyclass(name = "MandelstamTDistribution", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMandelstamTDistribution(pub MandelstamTDistribution);

#[pymethods]
impl PyMandelstamTDistribution {
    /// Construct an exponential Mandelstam-t distribution.
    #[staticmethod]
    fn exponential(slope: f64) -> PyResult<Self> {
        if slope <= 0.0 {
            return Err(PyValueError::new_err(
                "`slope` must be positive for an exponential distribution",
            ));
        }
        Ok(Self(MandelstamTDistribution::Exponential { slope }))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Generator settings for an initial-state particle.
#[pyclass(name = "GenInitialState", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGenInitialState(pub GenInitialState);

#[pymethods]
impl PyGenInitialState {
    /// Construct a beam with fixed energy.
    #[staticmethod]
    fn beam_with_fixed_energy(mass: f64, energy: f64) -> Self {
        Self(GenInitialState::beam_with_fixed_energy(mass, energy))
    }

    /// Construct a beam with uniformly sampled energy.
    #[staticmethod]
    fn beam(mass: f64, min_energy: f64, max_energy: f64) -> Self {
        Self(GenInitialState::beam(mass, min_energy, max_energy))
    }

    /// Construct a target at rest.
    #[staticmethod]
    fn target(mass: f64) -> Self {
        Self(GenInitialState::target(mass))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Generator settings for a generated composite particle.
#[pyclass(name = "GenComposite", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGenComposite(pub GenComposite);

#[pymethods]
impl PyGenComposite {
    /// Construct a composite mass generator with a uniform mass range.
    #[new]
    fn new(min_mass: f64, max_mass: f64) -> Self {
        Self(GenComposite::new(min_mass, max_mass))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Generator settings for a stable generated final-state particle.
#[pyclass(name = "GenFinalState", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGenFinalState(pub GenFinalState);

#[pymethods]
impl PyGenFinalState {
    /// Construct a fixed-mass final-state generator.
    #[new]
    fn new(mass: f64) -> Self {
        Self(GenFinalState::new(mass))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Reconstruction metadata for a generated particle.
#[pyclass(name = "Reconstruction", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyReconstruction(pub Reconstruction);

#[pymethods]
impl PyReconstruction {
    /// Mark a generated particle as stored in one or more dataset p4 columns.
    #[staticmethod]
    fn reconstructed(p4_names: Vec<String>) -> Self {
        Self(Reconstruction::Reconstructed { p4_names })
    }

    /// Mark a generated particle as fixed in the reconstructed reaction.
    #[staticmethod]
    fn fixed(p4: &PyVec4) -> Self {
        Self(Reconstruction::Fixed(p4.0))
    }

    /// Mark a generated particle as missing in the reconstructed reaction.
    #[staticmethod]
    fn missing() -> Self {
        Self(Reconstruction::Missing)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A generated initial-state particle.
#[pyclass(name = "InitialStateParticle", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyInitialStateParticle(pub InitialStateParticle);

#[pymethods]
impl PyInitialStateParticle {
    /// Construct a generated initial-state particle.
    #[new]
    fn new(label: &str, generator: &PyGenInitialState, reconstruction: &PyReconstruction) -> Self {
        Self(InitialStateParticle::new(
            label,
            generator.0.clone(),
            reconstruction.0.clone(),
        ))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A generated final-state particle or generated composite decay.
#[pyclass(name = "FinalStateParticle", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyFinalStateParticle(pub FinalStateParticle);

#[pymethods]
impl PyFinalStateParticle {
    /// Construct a generated stable final-state particle.
    #[new]
    fn new(label: &str, generator: &PyGenFinalState, reconstruction: &PyReconstruction) -> Self {
        Self(FinalStateParticle::new(
            label,
            generator.0.clone(),
            reconstruction.0.clone(),
        ))
    }

    /// Construct a generated composite from exactly two ordered daughters.
    #[staticmethod]
    fn composite(
        label: &str,
        generator: &PyGenComposite,
        daughters: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if daughters.len() != 2 {
            return Err(PyValueError::new_err(
                "composite particles require exactly two ordered daughters",
            ));
        }
        let daughter_1 = daughters.get_item(0)?.extract::<Self>()?;
        let daughter_2 = daughters.get_item(1)?.extract::<Self>()?;
        Ok(Self(FinalStateParticle::composite(
            label,
            generator.0.clone(),
            (&daughter_1.0, &daughter_2.0),
        )))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A generated reaction layout.
#[pyclass(name = "GenReaction", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGenReaction(pub GenReaction);

#[pymethods]
impl PyGenReaction {
    /// Construct a generated two-to-two reaction.
    #[staticmethod]
    fn two_to_two(
        p1: &PyInitialStateParticle,
        p2: &PyInitialStateParticle,
        p3: &PyFinalStateParticle,
        p4: &PyFinalStateParticle,
        tdist: &PyMandelstamTDistribution,
    ) -> Self {
        Self(GenReaction::two_to_two(
            p1.0.clone(),
            p2.0.clone(),
            p3.0.clone(),
            p4.0.clone(),
            tdist.0.clone(),
        ))
    }

    /// Return generated p4 labels.
    fn p4_labels(&self) -> Vec<String> {
        self.0.p4_labels()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Event generator for generated reaction layouts.
#[pyclass(name = "EventGenerator", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyEventGenerator(pub EventGenerator);

#[pymethods]
impl PyEventGenerator {
    /// Construct an event generator.
    #[new]
    #[pyo3(signature = (reaction, aux_generators=None, seed=None))]
    fn new(
        reaction: &PyGenReaction,
        aux_generators: Option<HashMap<String, PyDistribution>>,
        seed: Option<u64>,
    ) -> Self {
        Self(EventGenerator::new(
            reaction.0.clone(),
            aux_generators
                .unwrap_or_default()
                .into_iter()
                .map(|(name, distribution)| (name, distribution.0))
                .collect(),
            seed,
        ))
    }

    /// Generate a dataset.
    fn generate_dataset(&self, n_events: usize) -> PyResult<PyDataset> {
        Ok(PyDataset(Arc::new(self.0.generate_dataset(n_events)?)))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}
