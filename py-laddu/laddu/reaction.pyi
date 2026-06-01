from collections.abc import Sequence
from enum import Enum
from typing import Literal

from laddu.generation import MassSampler, MomentumSource, VertexGenerator
from laddu.quantum import ParticleProperties
from laddu.variables import Angles, Mandelstam, Mass, PolAngle, Polarization

class Axis:
    @staticmethod
    def particle(particle: str) -> Axis: ...
    @staticmethod
    def opposite(particle: str) -> Axis: ...
    @staticmethod
    def normal(a: str, b: str) -> Axis: ...
    def at(self, vertex: str) -> Axis: ...
    def flipped(self) -> Axis: ...

class Axes:
    @staticmethod
    def from_y_z(y: Axis, z: Axis) -> Axes: ...

class Frame:
    origin: str

    def __init__(self, origin: str, axes: Axes) -> None: ...

class ParticleSource(Enum):
    Inferred = 0
    Stored = 1
    Missing = 2

class Channel:
    def __init__(self) -> None: ...
    def create_vertex(
        self,
        label: str,
        incoming: Sequence[str],
        outgoing: Sequence[str],
        *,
        generator: VertexGenerator | None = None,
    ) -> None: ...
    def create_decay(
        self,
        label: str,
        parent: str,
        daughters: Sequence[str],
        *,
        generator: VertexGenerator | None = None,
    ) -> None: ...
    def create_production(
        self,
        label: str,
        incoming: Sequence[str],
        outgoing: Sequence[str],
        *,
        generator: VertexGenerator | None = None,
    ) -> None: ...
    def edit_particle(
        self,
        particle: str,
        *,
        source: ParticleSource | None = None,
        properties: ParticleProperties | None = None,
        mass: float | None = None,
        momentum: MomentumSource | None = None,
        mass_sampler: MassSampler | None = None,
        name: str | None = None,
        species: str | None = None,
        self_conjugate: bool | None = None,
    ) -> None: ...
    def edit_vertex(self, vertex: str, *, generator: VertexGenerator) -> None: ...
    def mass(self, particle: str) -> Mass: ...
    def angles(self, particle: str, frame: Frame) -> Angles: ...
    def mandelstam(
        self, vertex: str, channel: Literal['s', 't', 'u', 'S', 'T', 'U']
    ) -> Mandelstam: ...
    def pol_angle(self, vertex: str, angle_aux: str) -> PolAngle: ...
    def polarization(
        self, vertex: str, *, pol_magnitude: str, pol_angle: str
    ) -> Polarization: ...

__all__ = ['Axes', 'Axis', 'Channel', 'Frame', 'ParticleSource']
