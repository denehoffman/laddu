from collections.abc import Callable
from typing import Any, overload

from laddu.amplitude import Expression
from laddu.data import Dataset
from laddu.reaction import Channel
from laddu.vectors import Vec4

class MomentumSource: ...
class MassSampler: ...
class VertexGenerator: ...

class Envelope:
    @staticmethod
    def initial(value: float) -> Envelope: ...
    @staticmethod
    def estimate(pilot_events: int, safety_factor: float) -> Envelope: ...
    @staticmethod
    def adaptive(initial: float, growth_factor: float) -> Envelope: ...

class EnvelopeViolationPolicy:
    Error: EnvelopeViolationPolicy
    WarnAndContinue: EnvelopeViolationPolicy
    Grow: EnvelopeViolationPolicy

class GenerationMode:
    @staticmethod
    def raw() -> GenerationMode: ...
    @staticmethod
    def weighted(expression: Expression, parameters: list[float]) -> GenerationMode: ...
    @staticmethod
    def accepted(
        expression: Expression, parameters: list[float], envelope: Envelope | float
    ) -> GenerationMode: ...

class GenerationOutput:
    @staticmethod
    def all() -> GenerationOutput: ...
    @staticmethod
    def final_state() -> GenerationOutput: ...
    @staticmethod
    def only(labels: list[str]) -> GenerationOutput: ...
    @staticmethod
    def exclude(labels: list[str]) -> GenerationOutput: ...

class GenerationOptions:
    batch_size: int
    max_trials: int | None
    seed: int | None
    envelope_violation_policy: EnvelopeViolationPolicy

    def __init__(
        self,
        *,
        batch_size: int = 10_000,
        max_trials: int | None = None,
        seed: int | None = None,
        envelope_violation_policy: EnvelopeViolationPolicy | None = None,
    ) -> None: ...

class DatasetSink:
    def __init__(self, *, output: GenerationOutput | None = None) -> None: ...

class CallbackSink:
    def __init__(self, callback: Callable[[list[dict[str, Any]]], object]) -> None: ...

class ParquetSink:
    def __init__(
        self,
        path: str,
        *,
        output: GenerationOutput | None = None,
        batch_size: int = 10_000,
        precision: str | None = None,
    ) -> None: ...

class RootSink:
    def __init__(
        self,
        path: str,
        *,
        output: GenerationOutput | None = None,
        batch_size: int = 10_000,
        precision: str | None = None,
        tree: str | None = None,
    ) -> None: ...

class GeneratedSink: ...

class EnvelopeStats:
    configured_max: float | None
    pilot_events: int
    pilot_observed_max: float | None
    safety_factor: float | None
    growth_factor: float | None
    observed_max: float | None
    violations: int
    largest_violation_ratio: float | None
    updates: int
    final_max: float | None

class GenerationStats:
    target_events: int
    written_events: int
    proposed_events: int
    accepted_events: int
    rejected_events: int
    acceptance_rate: float | None
    envelope: float | None
    envelope_violations: int
    envelope_stats: EnvelopeStats | None
    sum_weights: float
    min_weight: float | None
    max_weight: float | None
    batches_written: int
    def audit(self) -> str: ...

class GenerationResult:
    output: object
    stats: GenerationStats

class DatasetGenerationResult(GenerationResult):
    output: Dataset

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
    @overload
    def generate(
        self,
        target_events: int,
        sink: DatasetSink,
        *,
        mode: GenerationMode | None = None,
        options: GenerationOptions | None = None,
    ) -> DatasetGenerationResult: ...
    @overload
    def generate(
        self,
        target_events: int,
        sink: GeneratedSink,
        *,
        mode: GenerationMode | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult: ...
    @overload
    def generate(
        self,
        target_events: int,
        sink: ParquetSink | RootSink | CallbackSink,
        *,
        mode: GenerationMode | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult: ...

__all__ = [
    'CallbackSink',
    'DatasetSink',
    'DecayParticlePlan',
    'DecayPlan',
    'Envelope',
    'EnvelopeStats',
    'EnvelopeViolationPolicy',
    'EventGenerator',
    'GeneratedEvent',
    'GeneratedSink',
    'GenerationMode',
    'GenerationOptions',
    'GenerationOutput',
    'GenerationPlan',
    'GenerationResult',
    'GenerationStats',
    'InitialParticlePlan',
    'MassSampler',
    'MomentumSource',
    'ParquetSink',
    'PlannedMass',
    'ProductionPlan',
    'RootSink',
    'VertexGenerator',
]
