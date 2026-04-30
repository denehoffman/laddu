from collections.abc import Mapping

from laddu.data import Dataset
from laddu.math import Histogram
from laddu.reaction import Reaction
from laddu.vectors import Vec4

class Distribution:
    @staticmethod
    def fixed(value: float) -> Distribution: ...
    @staticmethod
    def uniform(min: float, max: float) -> Distribution: ...
    @staticmethod
    def normal(mu: float, sigma: float) -> Distribution: ...
    @staticmethod
    def exponential(slope: float) -> Distribution: ...
    @staticmethod
    def histogram(histogram: Histogram) -> Distribution: ...

class MandelstamTDistribution:
    @staticmethod
    def exponential(slope: float) -> MandelstamTDistribution: ...
    @staticmethod
    def histogram(histogram: Histogram) -> MandelstamTDistribution: ...

class InitialGenerator:
    @staticmethod
    def beam_with_fixed_energy(mass: float, energy: float) -> InitialGenerator: ...
    @staticmethod
    def beam(mass: float, min_energy: float, max_energy: float) -> InitialGenerator: ...
    @staticmethod
    def beam_with_energy_histogram(
        mass: float, energy: Histogram
    ) -> InitialGenerator: ...
    @staticmethod
    def target(mass: float) -> InitialGenerator: ...

class CompositeGenerator:
    def __init__(self, min_mass: float, max_mass: float) -> None: ...

class StableGenerator:
    def __init__(self, mass: float) -> None: ...

class Reconstruction:
    @staticmethod
    def stored() -> Reconstruction: ...
    @staticmethod
    def fixed(p4: Vec4) -> Reconstruction: ...
    @staticmethod
    def missing() -> Reconstruction: ...
    @staticmethod
    def composite() -> Reconstruction: ...

class ParticleSpecies:
    id: int | None
    namespace: str | None
    label_value: str | None

    @staticmethod
    def code(id: int) -> ParticleSpecies: ...
    @staticmethod
    def with_namespace(namespace: str, id: int) -> ParticleSpecies: ...
    @staticmethod
    def label(label: str) -> ParticleSpecies: ...

class GeneratedParticle:
    id: str
    species: ParticleSpecies | None

    @staticmethod
    def initial(
        id: str,
        generator: InitialGenerator,
        reconstruction: Reconstruction,
    ) -> GeneratedParticle: ...
    @staticmethod
    def stable(
        id: str,
        generator: StableGenerator,
        reconstruction: Reconstruction,
    ) -> GeneratedParticle: ...
    @staticmethod
    def composite(
        id: str,
        generator: CompositeGenerator,
        daughters: tuple[GeneratedParticle, GeneratedParticle],
        reconstruction: Reconstruction,
    ) -> GeneratedParticle: ...
    def with_species(self, species: ParticleSpecies) -> GeneratedParticle: ...

class GeneratedReaction:
    @staticmethod
    def two_to_two(
        p1: GeneratedParticle,
        p2: GeneratedParticle,
        p3: GeneratedParticle,
        p4: GeneratedParticle,
        tdist: MandelstamTDistribution,
    ) -> GeneratedReaction: ...
    def p4_labels(self) -> list[str]: ...
    def particle_layouts(self) -> list[GeneratedParticleLayout]: ...
    def reconstructed_reaction(self) -> Reaction: ...

class GeneratedStorage:
    @staticmethod
    def all() -> GeneratedStorage: ...
    @staticmethod
    def only(ids: list[str] | tuple[str, ...]) -> GeneratedStorage: ...

class GeneratedParticleLayout:
    id: str
    product_id: int
    parent_id: int | None
    species: ParticleSpecies | None
    p4_label: str | None
    produced_vertex_id: int | None
    decay_vertex_id: int | None

class GeneratedVertexLayout:
    vertex_id: int
    kind: str
    incoming_product_ids: list[int]
    outgoing_product_ids: list[int]

class GeneratedEventLayout:
    p4_labels: list[str]
    aux_labels: list[str]
    particles: list[GeneratedParticleLayout]
    vertices: list[GeneratedVertexLayout]

class GeneratedBatch:
    dataset: Dataset
    reaction: GeneratedReaction
    layout: GeneratedEventLayout

class GeneratedBatchIter:
    def __iter__(self) -> GeneratedBatchIter: ...
    def __next__(self) -> GeneratedBatch: ...

class EventGenerator:
    def __init__(
        self,
        reaction: GeneratedReaction,
        aux_generators: Mapping[str, Distribution] | None = None,
        seed: int | None = None,
        storage: GeneratedStorage | None = None,
    ) -> None: ...
    def generate_batch(self, n_events: int) -> GeneratedBatch: ...
    def generate_batches(
        self, total_events: int, batch_size: int
    ) -> GeneratedBatchIter: ...
    def generate_dataset(self, n_events: int) -> Dataset: ...
