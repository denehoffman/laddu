from collections.abc import Mapping

from laddu.data import Dataset
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

class GenInitialState:
    @staticmethod
    def beam_with_fixed_energy(mass: float, energy: float) -> GenInitialState: ...
    @staticmethod
    def beam(mass: float, min_energy: float, max_energy: float) -> GenInitialState: ...
    @staticmethod
    def target(mass: float) -> GenInitialState: ...

class GenComposite:
    def __init__(self, min_mass: float, max_mass: float) -> None: ...

class GenFinalState:
    def __init__(self, mass: float) -> None: ...

class Reconstruction:
    @staticmethod
    def reconstructed(p4_names: list[str]) -> Reconstruction: ...
    @staticmethod
    def fixed(p4: Vec4) -> Reconstruction: ...
    @staticmethod
    def missing() -> Reconstruction: ...

class InitialStateParticle:
    def __init__(
        self,
        label: str,
        generator: GenInitialState,
        reconstruction: Reconstruction,
    ) -> None: ...

class FinalStateParticle:
    def __init__(
        self,
        label: str,
        generator: GenFinalState,
        reconstruction: Reconstruction,
    ) -> None: ...
    @staticmethod
    def composite(
        label: str,
        generator: GenComposite,
        daughters: tuple[FinalStateParticle, FinalStateParticle],
    ) -> FinalStateParticle: ...

class GenReaction:
    @staticmethod
    def two_to_two(
        p1: InitialStateParticle,
        p2: InitialStateParticle,
        p3: FinalStateParticle,
        p4: FinalStateParticle,
        tdist: MandelstamTDistribution,
    ) -> GenReaction: ...
    def p4_labels(self) -> list[str]: ...

class EventGenerator:
    def __init__(
        self,
        reaction: GenReaction,
        aux_generators: Mapping[str, Distribution] | None = None,
        seed: int | None = None,
    ) -> None: ...
    def generate_dataset(self, n_events: int) -> Dataset: ...
