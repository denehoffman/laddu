from collections.abc import Mapping

from laddu.data import Dataset
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

class MandelstamTDistribution:
    @staticmethod
    def exponential(slope: float) -> MandelstamTDistribution: ...

class InitialGenerator:
    @staticmethod
    def beam_with_fixed_energy(mass: float, energy: float) -> InitialGenerator: ...
    @staticmethod
    def beam(mass: float, min_energy: float, max_energy: float) -> InitialGenerator: ...
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

class GeneratedParticle:
    id: str

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

class GeneratedParticleLayout:
    id: str
    product_id: int
    parent_id: int | None
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

class EventGenerator:
    def __init__(
        self,
        reaction: GeneratedReaction,
        aux_generators: Mapping[str, Distribution] | None = None,
        seed: int | None = None,
    ) -> None: ...
    def generate_batch(self, n_events: int) -> GeneratedBatch: ...
    def generate_dataset(self, n_events: int) -> Dataset: ...
