from laddu.data import Dataset
from laddu.reaction import Channel
from laddu.vectors import Vec4

class MomentumSource: ...
class MassSampler: ...
class VertexGenerator: ...

class PlannedMass:
    kind: str
    value: float | None

class InitialParticlePlan:
    label: str
    mass: float
    momentum: MomentumSource

class DecayParticlePlan:
    label: str
    mass: PlannedMass
    decay: DecayPlan | None

class DecayPlan:
    vertex: str
    daughters: list[DecayParticlePlan]

class ProductionPlan:
    vertex: str
    incoming: list[InitialParticlePlan]
    outgoing: list[DecayParticlePlan]

class GenerationPlan:
    production: ProductionPlan

    def __init__(self, channel: Channel) -> None: ...
    @staticmethod
    def from_channel(channel: Channel) -> GenerationPlan: ...

class GeneratedEvent:
    def labels(self) -> list[str]: ...
    def p4(self, label: str) -> Vec4 | None: ...
    def p4s(self) -> list[tuple[str, Vec4]]: ...

class EventGenerator:
    plan: GenerationPlan

    def __init__(self, channel: Channel, *, seed: int | None = None) -> None: ...
    @staticmethod
    def from_channel(channel: Channel, *, seed: int | None = None) -> EventGenerator: ...
    def with_seed(self, seed: int) -> EventGenerator: ...
    def p4_labels(self) -> list[str]: ...
    def generate_event(self) -> GeneratedEvent: ...
    def generate_dataset(self, n_events: int) -> Dataset: ...

__all__ = [
    'DecayParticlePlan',
    'DecayPlan',
    'EventGenerator',
    'GeneratedEvent',
    'GenerationPlan',
    'InitialParticlePlan',
    'MassSampler',
    'MomentumSource',
    'PlannedMass',
    'ProductionPlan',
    'VertexGenerator',
]
