from collections.abc import Sequence
from typing import Literal, TypeAlias

from laddu.amplitude import Expression
from laddu.quantum import QuantumNumber
from laddu.variables import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
)
from laddu.vectors import Vec4

_P4Selection: TypeAlias = str | Sequence[str]
_Frame: TypeAlias = Literal[
    'Helicity',
    'HX',
    'HEL',
    'GottfriedJackson',
    'Gottfried Jackson',
    'GJ',
    'Gottfried-Jackson',
    'Canonical',
    'CanonicalHelicity',
    'CH',
    'Adair',
    'AD',
]

class Particle:
    label: str

    @staticmethod
    def measured(label: str, p4: _P4Selection) -> Particle: ...
    @staticmethod
    def fixed(label: str, p4: Vec4) -> Particle: ...
    @staticmethod
    def missing(label: str) -> Particle: ...
    @staticmethod
    def composite(label: str, daughters: Sequence[Particle]) -> Particle: ...

class Decay:
    parent: Particle
    daughter_1: Particle
    daughter_2: Particle

    def daughters(self) -> list[Particle]: ...
    def mass(self) -> Mass: ...
    def parent_mass(self) -> Mass: ...
    def daughter_1_mass(self) -> Mass: ...
    def daughter_2_mass(self) -> Mass: ...
    def daughter_mass(self, daughter: Particle) -> Mass: ...
    def costheta(self, daughter: Particle, frame: _Frame = 'Helicity') -> CosTheta: ...
    def phi(self, daughter: Particle, frame: _Frame = 'Helicity') -> Phi: ...
    def angles(self, daughter: Particle, frame: _Frame = 'Helicity') -> Angles: ...
    def helicity_factor(
        self,
        name: str,
        spin: QuantumNumber,
        projection: QuantumNumber,
        daughter: Particle,
        lambda_1: QuantumNumber,
        lambda_2: QuantumNumber,
        frame: _Frame = 'Helicity',
    ) -> Expression: ...
    def canonical_factor(
        self,
        name: str,
        spin: QuantumNumber,
        projection: QuantumNumber,
        orbital_l: QuantumNumber,
        coupled_spin: QuantumNumber,
        daughter: Particle,
        daughter_1_spin: QuantumNumber,
        daughter_2_spin: QuantumNumber,
        lambda_1: QuantumNumber,
        lambda_2: QuantumNumber,
        frame: _Frame = 'Helicity',
    ) -> Expression: ...

class Reaction:
    @staticmethod
    def two_to_two(
        p1: Particle, p2: Particle, p3: Particle, p4: Particle
    ) -> Reaction: ...
    def mass(self, particle: Particle) -> Mass: ...
    def decay(self, parent: Particle) -> Decay: ...
    def mandelstam(
        self, channel: Literal['s', 't', 'u', 'S', 'T', 'U']
    ) -> Mandelstam: ...
    def pol_angle(self, pol_angle: str) -> PolAngle: ...
    def polarization(self, pol_magnitude: str, pol_angle: str) -> Polarization: ...

__all__ = ['Decay', 'Particle', 'Reaction']
