use std::{collections::HashMap, sync::Arc};

use laddu_generation::{
    CompositeGenerator, Distribution, EventGenerator, GeneratedParticle, GeneratedReaction,
    InitialGenerator, MandelstamTDistribution, Reconstruction, StableGenerator,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};

use crate::{data::PyDataset, variables::PyReaction, vectors::PyVec4};

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

/// Generator settings for an initial generated particle.
#[pyclass(name = "InitialGenerator", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyInitialGenerator(pub InitialGenerator);

#[pymethods]
impl PyInitialGenerator {
    /// Construct a beam with fixed energy.
    #[staticmethod]
    fn beam_with_fixed_energy(mass: f64, energy: f64) -> Self {
        Self(InitialGenerator::beam_with_fixed_energy(mass, energy))
    }

    /// Construct a beam with uniformly sampled energy.
    #[staticmethod]
    fn beam(mass: f64, min_energy: f64, max_energy: f64) -> Self {
        Self(InitialGenerator::beam(mass, min_energy, max_energy))
    }

    /// Construct a target at rest.
    #[staticmethod]
    fn target(mass: f64) -> Self {
        Self(InitialGenerator::target(mass))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Generator settings for a generated composite particle.
#[pyclass(name = "CompositeGenerator", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCompositeGenerator(pub CompositeGenerator);

#[pymethods]
impl PyCompositeGenerator {
    /// Construct a composite mass generator with a uniform mass range.
    #[new]
    fn new(min_mass: f64, max_mass: f64) -> Self {
        Self(CompositeGenerator::new(min_mass, max_mass))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Generator settings for a stable generated particle.
#[pyclass(name = "StableGenerator", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyStableGenerator(pub StableGenerator);

#[pymethods]
impl PyStableGenerator {
    /// Construct a fixed-mass stable-particle generator.
    #[new]
    fn new(mass: f64) -> Self {
        Self(StableGenerator::new(mass))
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
    /// Mark a generated particle as stored under its generated ID.
    #[staticmethod]
    fn stored() -> Self {
        Self(Reconstruction::Stored)
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

    /// Mark a generated particle as reconstructed from its generated daughters.
    #[staticmethod]
    fn composite() -> Self {
        Self(Reconstruction::Composite)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A generated particle with generation and reconstruction metadata.
#[pyclass(name = "GeneratedParticle", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGeneratedParticle(pub GeneratedParticle);

#[pymethods]
impl PyGeneratedParticle {
    /// Construct an initial generated particle.
    #[staticmethod]
    fn initial(
        id: &str,
        generator: &PyInitialGenerator,
        reconstruction: &PyReconstruction,
    ) -> Self {
        Self(GeneratedParticle::initial(
            id,
            generator.0.clone(),
            reconstruction.0.clone(),
        ))
    }

    /// Construct a stable generated particle.
    #[staticmethod]
    fn stable(id: &str, generator: &PyStableGenerator, reconstruction: &PyReconstruction) -> Self {
        Self(GeneratedParticle::stable(
            id,
            generator.0.clone(),
            reconstruction.0.clone(),
        ))
    }

    /// Construct a generated composite from exactly two ordered daughters.
    #[staticmethod]
    fn composite(
        id: &str,
        generator: &PyCompositeGenerator,
        daughters: &Bound<'_, PyTuple>,
        reconstruction: &PyReconstruction,
    ) -> PyResult<Self> {
        if daughters.len() != 2 {
            return Err(PyValueError::new_err(
                "composite particles require exactly two ordered daughters",
            ));
        }
        let daughter_1 = daughters.get_item(0)?.extract::<Self>()?;
        let daughter_2 = daughters.get_item(1)?.extract::<Self>()?;
        Ok(Self(GeneratedParticle::composite(
            id,
            generator.0.clone(),
            (&daughter_1.0, &daughter_2.0),
            reconstruction.0.clone(),
        )))
    }

    /// The generated particle ID.
    #[getter]
    fn id(&self) -> String {
        self.0.id().to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A generated reaction layout.
#[pyclass(name = "GeneratedReaction", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGeneratedReaction(pub GeneratedReaction);

#[pymethods]
impl PyGeneratedReaction {
    /// Construct a generated two-to-two reaction.
    #[staticmethod]
    fn two_to_two(
        p1: &PyGeneratedParticle,
        p2: &PyGeneratedParticle,
        p3: &PyGeneratedParticle,
        p4: &PyGeneratedParticle,
        tdist: &PyMandelstamTDistribution,
    ) -> PyResult<Self> {
        Ok(Self(GeneratedReaction::two_to_two(
            p1.0.clone(),
            p2.0.clone(),
            p3.0.clone(),
            p4.0.clone(),
            tdist.0.clone(),
        )?))
    }

    /// Return generated p4 labels.
    fn p4_labels(&self) -> Vec<String> {
        self.0.p4_labels()
    }

    /// Build the reconstructed reaction corresponding to this generated layout.
    fn reconstructed_reaction(&self) -> PyResult<PyReaction> {
        Ok(PyReaction(self.0.reconstructed_reaction()?))
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
        reaction: &PyGeneratedReaction,
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
