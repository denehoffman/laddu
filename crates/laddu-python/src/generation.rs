use std::{collections::HashMap, sync::Arc};

use laddu_generation::{
    CompositeGenerator, Distribution, EventGenerator, GeneratedBatch, GeneratedEventLayout,
    GeneratedParticle, GeneratedParticleLayout, GeneratedReaction, GeneratedStorage,
    GeneratedVertexKind, GeneratedVertexLayout, HistogramSampler, InitialGenerator,
    MandelstamTDistribution, ParticleSpecies, Reconstruction, StableGenerator,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyTuple};

use crate::{data::PyDataset, math::PyHistogram, variables::PyReaction, vectors::PyVec4};

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

    /// Construct a histogram-sampled scalar distribution.
    #[staticmethod]
    fn histogram(histogram: &PyHistogram) -> PyResult<Self> {
        Ok(Self(Distribution::Histogram(HistogramSampler::new(
            histogram.0.clone(),
        )?)))
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

    /// Construct a histogram-sampled Mandelstam-t distribution.
    #[staticmethod]
    fn histogram(histogram: &PyHistogram) -> PyResult<Self> {
        Ok(Self(MandelstamTDistribution::Histogram(
            HistogramSampler::new(histogram.0.clone())?,
        )))
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

    /// Construct a beam with histogram-sampled energy.
    #[staticmethod]
    fn beam_with_energy_histogram(mass: f64, energy: &PyHistogram) -> PyResult<Self> {
        Ok(Self(InitialGenerator::beam_with_energy_histogram(
            mass,
            energy.0.clone(),
        )?))
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

/// Experiment-neutral metadata describing a generated particle species.
#[pyclass(name = "ParticleSpecies", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyParticleSpecies(pub ParticleSpecies);

#[pymethods]
impl PyParticleSpecies {
    /// Construct a species from a numeric code with no namespace.
    #[staticmethod]
    fn code(id: i64) -> Self {
        Self(ParticleSpecies::code(id))
    }

    /// Construct a species from a numeric code in an explicit namespace.
    #[staticmethod]
    fn with_namespace(namespace: &str, id: i64) -> Self {
        Self(ParticleSpecies::with_namespace(namespace, id))
    }

    /// Construct a species from a free-form label.
    #[staticmethod]
    fn label(label: &str) -> Self {
        Self(ParticleSpecies::label(label))
    }

    /// The numeric species code, if this is a code-based species.
    #[getter]
    fn id(&self) -> Option<i64> {
        match &self.0 {
            ParticleSpecies::Code { id, .. } => Some(*id),
            ParticleSpecies::Label(_) => None,
        }
    }

    /// The numeric species namespace, if this is a namespaced code-based species.
    #[getter]
    fn namespace(&self) -> Option<String> {
        match &self.0 {
            ParticleSpecies::Code { namespace, .. } => namespace.clone(),
            ParticleSpecies::Label(_) => None,
        }
    }

    /// The species label, if this is a label-based species.
    #[getter]
    fn label_value(&self) -> Option<String> {
        match &self.0 {
            ParticleSpecies::Code { .. } => None,
            ParticleSpecies::Label(label) => Some(label.clone()),
        }
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

    /// Return a copy of this generated particle with species metadata attached.
    fn with_species(&self, species: &PyParticleSpecies) -> Self {
        Self(self.0.clone().with_species(species.0.clone()))
    }

    /// The generated particle ID.
    #[getter]
    fn id(&self) -> String {
        self.0.id().to_string()
    }

    /// Optional species metadata for this generated particle.
    #[getter]
    fn species(&self) -> Option<PyParticleSpecies> {
        self.0.species().cloned().map(PyParticleSpecies)
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

    /// Return generated particle layout entries in stable product-ID order.
    fn particle_layouts(&self) -> Vec<PyGeneratedParticleLayout> {
        self.0
            .particle_layouts()
            .into_iter()
            .map(PyGeneratedParticleLayout)
            .collect()
    }

    /// Build the reconstructed reaction corresponding to this generated layout.
    fn reconstructed_reaction(&self) -> PyResult<PyReaction> {
        Ok(PyReaction(self.0.reconstructed_reaction()?))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Selects which generated particle p4s are written into generated datasets.
#[pyclass(name = "GeneratedStorage", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGeneratedStorage(pub GeneratedStorage);

#[pymethods]
impl PyGeneratedStorage {
    /// Store every generated particle p4.
    #[staticmethod]
    fn all() -> Self {
        Self(GeneratedStorage::all())
    }

    /// Store only the listed generated particle IDs.
    #[staticmethod]
    fn only(ids: Vec<String>) -> Self {
        Self(GeneratedStorage::only(ids))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Metadata for one generated particle in a generated event layout.
#[pyclass(name = "GeneratedParticleLayout", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGeneratedParticleLayout(pub GeneratedParticleLayout);

#[pymethods]
impl PyGeneratedParticleLayout {
    /// The generated particle identifier.
    #[getter]
    fn id(&self) -> String {
        self.0.id().to_string()
    }

    /// The zero-based stable product ID in generated-layout order.
    #[getter]
    fn product_id(&self) -> usize {
        self.0.product_id()
    }

    /// The decay-parent product ID, or None if this particle has no decay parent.
    #[getter]
    fn parent_id(&self) -> Option<usize> {
        self.0.parent_id()
    }

    /// Optional species metadata associated with this generated particle.
    #[getter]
    fn species(&self) -> Option<PyParticleSpecies> {
        self.0.species().cloned().map(PyParticleSpecies)
    }

    /// The dataset p4 label associated with this particle, if stored in the batch.
    #[getter]
    fn p4_label(&self) -> Option<String> {
        self.0.p4_label().map(str::to_string)
    }

    /// The vertex ID where this particle was produced, if any.
    #[getter]
    fn produced_vertex_id(&self) -> Option<usize> {
        self.0.produced_vertex_id()
    }

    /// The vertex ID where this particle decays, if it is a generated parent.
    #[getter]
    fn decay_vertex_id(&self) -> Option<usize> {
        self.0.decay_vertex_id()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Metadata for one generated vertex in a generated event layout.
#[pyclass(name = "GeneratedVertexLayout", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGeneratedVertexLayout(pub GeneratedVertexLayout);

#[pymethods]
impl PyGeneratedVertexLayout {
    /// The zero-based stable vertex ID in generated-layout order.
    #[getter]
    fn vertex_id(&self) -> usize {
        self.0.vertex_id()
    }

    /// The semantic vertex kind.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.0.kind() {
            GeneratedVertexKind::Production => "Production",
            GeneratedVertexKind::Decay => "Decay",
        }
    }

    /// Product IDs entering this vertex.
    #[getter]
    fn incoming_product_ids(&self) -> Vec<usize> {
        self.0.incoming_product_ids().to_vec()
    }

    /// Product IDs leaving this vertex.
    #[getter]
    fn outgoing_product_ids(&self) -> Vec<usize> {
        self.0.outgoing_product_ids().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Metadata describing the columns in a generated event batch.
#[pyclass(name = "GeneratedEventLayout", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGeneratedEventLayout(pub GeneratedEventLayout);

#[pymethods]
impl PyGeneratedEventLayout {
    /// Generated p4 column labels in dataset order.
    #[getter]
    fn p4_labels(&self) -> Vec<String> {
        self.0.p4_labels().to_vec()
    }

    /// Generated auxiliary column labels in dataset order.
    #[getter]
    fn aux_labels(&self) -> Vec<String> {
        self.0.aux_labels().to_vec()
    }

    /// Generated particle layout entries in stable product-ID order.
    #[getter]
    fn particles(&self) -> Vec<PyGeneratedParticleLayout> {
        self.0
            .particles()
            .iter()
            .cloned()
            .map(PyGeneratedParticleLayout)
            .collect()
    }

    /// Return the generated particle layout for a generated particle ID.
    fn particle(&self, id: &str) -> Option<PyGeneratedParticleLayout> {
        self.0.particle(id).cloned().map(PyGeneratedParticleLayout)
    }

    /// Return the generated particle layout for a stable product ID.
    fn product(&self, product_id: usize) -> Option<PyGeneratedParticleLayout> {
        self.0
            .product(product_id)
            .cloned()
            .map(PyGeneratedParticleLayout)
    }

    /// Generated vertex layout entries in stable vertex-ID order.
    #[getter]
    fn vertices(&self) -> Vec<PyGeneratedVertexLayout> {
        self.0
            .vertices()
            .iter()
            .cloned()
            .map(PyGeneratedVertexLayout)
            .collect()
    }

    /// Return the generated vertex layout for a stable vertex ID.
    fn vertex(&self, vertex_id: usize) -> Option<PyGeneratedVertexLayout> {
        self.0
            .vertex(vertex_id)
            .cloned()
            .map(PyGeneratedVertexLayout)
    }

    /// Return the production vertex layout, if the generated layout has one.
    fn production_vertex(&self) -> Option<PyGeneratedVertexLayout> {
        self.0
            .production_vertex()
            .cloned()
            .map(PyGeneratedVertexLayout)
    }

    /// Return the generated decay daughters of a parent product ID.
    fn decay_products(&self, parent_product_id: usize) -> Vec<PyGeneratedParticleLayout> {
        self.0
            .decay_products(parent_product_id)
            .into_iter()
            .cloned()
            .map(PyGeneratedParticleLayout)
            .collect()
    }

    /// Return production-level incoming particle layouts.
    fn production_incoming(&self) -> Vec<PyGeneratedParticleLayout> {
        self.0
            .production_incoming()
            .into_iter()
            .cloned()
            .map(PyGeneratedParticleLayout)
            .collect()
    }

    /// Return production-level outgoing particle layouts.
    fn production_outgoing(&self) -> Vec<PyGeneratedParticleLayout> {
        self.0
            .production_outgoing()
            .into_iter()
            .cloned()
            .map(PyGeneratedParticleLayout)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A generated dataset batch plus generated reaction and layout metadata.
#[pyclass(name = "GeneratedBatch", module = "laddu", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyGeneratedBatch(pub GeneratedBatch);

#[pymethods]
impl PyGeneratedBatch {
    /// The generated dataset for this batch.
    #[getter]
    fn dataset(&self) -> PyDataset {
        PyDataset(Arc::new(self.0.dataset().clone()))
    }

    /// The generated reaction metadata for this batch.
    #[getter]
    fn reaction(&self) -> PyGeneratedReaction {
        PyGeneratedReaction(self.0.reaction().clone())
    }

    /// The generated event layout metadata for this batch.
    #[getter]
    fn layout(&self) -> PyGeneratedEventLayout {
        PyGeneratedEventLayout(self.0.layout().clone())
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Finite iterator over generated dataset batches.
#[pyclass(
    name = "GeneratedBatchIter",
    module = "laddu",
    unsendable,
    skip_from_py_object
)]
pub struct PyGeneratedBatchIter {
    iter: Box<dyn Iterator<Item = laddu_core::LadduResult<GeneratedBatch>>>,
}

#[pymethods]
impl PyGeneratedBatchIter {
    fn __iter__(slf: PyRef<'_, Self>) -> Py<PyGeneratedBatchIter> {
        slf.into()
    }

    fn __next__(&mut self) -> PyResult<Option<PyGeneratedBatch>> {
        match self.iter.next() {
            Some(Ok(batch)) => Ok(Some(PyGeneratedBatch(batch))),
            Some(Err(err)) => Err(PyErr::from(err)),
            None => Ok(None),
        }
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
    #[pyo3(signature = (reaction, aux_generators=None, seed=None, storage=None))]
    fn new(
        reaction: &PyGeneratedReaction,
        aux_generators: Option<HashMap<String, PyDistribution>>,
        seed: Option<u64>,
        storage: Option<&PyGeneratedStorage>,
    ) -> PyResult<Self> {
        let generator = EventGenerator::new(
            reaction.0.clone(),
            aux_generators
                .unwrap_or_default()
                .into_iter()
                .map(|(name, distribution)| (name, distribution.0))
                .collect(),
            seed,
        );
        let generator = if let Some(storage) = storage {
            generator.with_storage(storage.0.clone())?
        } else {
            generator
        };
        Ok(Self(generator))
    }

    /// Generate one dataset batch with generated layout metadata.
    fn generate_batch(&self, n_events: usize) -> PyResult<PyGeneratedBatch> {
        Ok(PyGeneratedBatch(self.0.generate_batch(n_events)?))
    }

    /// Generate a finite iterator over generated dataset batches.
    fn generate_batches(
        &self,
        total_events: usize,
        batch_size: usize,
    ) -> PyResult<PyGeneratedBatchIter> {
        Ok(PyGeneratedBatchIter {
            iter: Box::new(self.0.generate_batches(total_events, batch_size)?),
        })
    }

    /// Generate a dataset.
    fn generate_dataset(&self, n_events: usize) -> PyResult<PyDataset> {
        Ok(PyDataset(Arc::new(self.0.generate_dataset(n_events)?)))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}
