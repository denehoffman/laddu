from laddu.data import Dataset
from laddu.reaction import Channel
from laddu.vectors import Vec4

class MomentumSource: ...
class MassSampler: ...
class VertexGenerator: ...
class Raw: ...

class GenerationOptions:
    batch_size: int
    max_trials: int | None
    seed: int | None

    def __init__(
        self,
        *,
        batch_size: int = 10_000,
        max_trials: int | None = None,
        seed: int | None = None,
    ) -> None: ...

class DatasetSink: ...

class GenerationStats:
    target_events: int
    written_events: int
    proposed_events: int
    accepted_events: int
    rejected_events: int
    acceptance_rate: float | None
    envelope: float | None
    envelope_violations: int
    batches_written: int
    def audit(self) -> str: ...

class GenerationResult:
    output: Dataset
    stats: GenerationStats

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
    def generate(
        self,
        target_events: int,
        sink: DatasetSink,
        *,
        mode: Raw | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult: ...

__all__ = [
    'DatasetSink',
    'DecayParticlePlan',
    'DecayPlan',
    'EventGenerator',
    'GeneratedEvent',
    'GenerationOptions',
    'GenerationPlan',
    'GenerationResult',
    'GenerationStats',
    'InitialParticlePlan',
    'MassSampler',
    'MomentumSource',
    'PlannedMass',
    'ProductionPlan',
    'Raw',
    'VertexGenerator',
]
