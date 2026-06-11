from collections.abc import Sequence
from enum import Enum
from fractions import Fraction
from typing import Literal

from laddu.generation import MassSampler, MomentumSource, VertexGenerator
from laddu.quantum import (
    Charge,
    Isospin,
    JLike,
    LLike,
    Parity,
    PartialWave,
    ParticleProperties,
    RuleSet,
    ScalarQuantumNumber,
    Sign,
    Statistics,
)
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

class Particle:
    label: str
    source: ParticleSource
    from_endpoint: str
    to_endpoint: str
    properties: ParticleProperties
    mass_sampler: MassSampler
    momentum: MomentumSource | None

class Vertex:
    label: str
    rules: RuleSet
    generation: VertexGenerator | None

class TwoBodyCoupling:
    parent_properties: ParticleProperties
    wave: PartialWave
    j: int | Fraction
    l: int
    s: int | Fraction

class Channel:
    def __init__(self) -> None: ...
    def create_vertex(
        self,
        label: str,
        incoming: Sequence[str],
        outgoing: Sequence[str],
        *,
        generator: VertexGenerator | None = None,
        rules: RuleSet | str | None = None,
    ) -> None: ...
    def create_decay(
        self,
        label: str,
        parent: str,
        daughters: Sequence[str],
        *,
        generator: VertexGenerator | None = None,
        rules: RuleSet | str | None = None,
    ) -> None: ...
    def create_production(
        self,
        label: str,
        incoming: Sequence[str],
        outgoing: Sequence[str],
        *,
        generator: VertexGenerator | None = None,
        rules: RuleSet | str | None = None,
    ) -> None: ...
    def edit_particle(
        self,
        particle: str,
        *,
        source: ParticleSource | None = None,
        name: str | None = None,
        species: str | None = None,
        antiparticle_species: str | None = None,
        self_conjugate: bool | None = None,
        mass: float | None = None,
        spin: JLike | None = None,
        parity: Parity | Sign | None = None,
        c_parity: Parity | Sign | None = None,
        g_parity: Parity | Sign | None = None,
        charge: Charge | ScalarQuantumNumber | None = None,
        isospin: Isospin | None = None,
        strangeness: int | None = None,
        charm: int | None = None,
        bottomness: int | None = None,
        topness: int | None = None,
        baryon_number: int | None = None,
        electron_lepton_number: int | None = None,
        muon_lepton_number: int | None = None,
        tau_lepton_number: int | None = None,
        statistics: Statistics | str | None = None,
        momentum: MomentumSource | None = None,
        mass_sampler: MassSampler | None = None,
        properties: ParticleProperties | None = None,
    ) -> None: ...
    def edit_vertex(
        self,
        vertex: str,
        *,
        generator: VertexGenerator | None = None,
        rules: RuleSet | str | None = None,
    ) -> None: ...
    def two_body_couplings(
        self, vertex: str, *, j_max: JLike, l_max: LLike
    ) -> list[TwoBodyCoupling]: ...
    def particles(self) -> list[Particle]: ...
    def vertices(self) -> list[Vertex]: ...
    def particle(self, particle: str) -> Particle: ...
    def vertex(self, vertex: str) -> Vertex: ...
    def incoming_particles(self, vertex: str) -> list[Particle]: ...
    def outgoing_particles(self, vertex: str) -> list[Particle]: ...
    def decay_vertices(self, particle: str) -> list[Vertex]: ...
    def mass(self, particle: str) -> Mass: ...
    def angles(self, particle: str, frame: Frame) -> Angles: ...
    def mandelstam(
        self, vertex: str, channel: Literal['s', 't', 'u', 'S', 'T', 'U']
    ) -> Mandelstam: ...
    def pol_angle(self, vertex: str, angle_aux: str) -> PolAngle: ...
    def polarization(
        self, vertex: str, *, pol_magnitude: str, pol_angle: str
    ) -> Polarization: ...

__all__ = [
    'Axes',
    'Axis',
    'Channel',
    'Frame',
    'Particle',
    'ParticleSource',
    'TwoBodyCoupling',
    'Vertex',
]
