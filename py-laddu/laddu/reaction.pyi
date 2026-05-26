from typing import Literal, TypeAlias

from laddu.amplitude import Expression
from laddu.quantum import JLike, LLike, MLike
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
    def stored(id: str) -> Particle: ...
    @staticmethod
    def fixed(label: str, p4: Vec4) -> Particle: ...
    @staticmethod
    def missing(label: str) -> Particle: ...
    @staticmethod
    def composite(label: str, daughters: tuple[Particle, Particle]) -> Particle: ...

class Decay:
    reaction: Reaction
    parent: str
    daughter_1: str
    daughter_2: str

    def daughters(self) -> list[str]: ...
    def mass(self) -> Mass: ...
    def parent_mass(self) -> Mass: ...
    def daughter_1_mass(self) -> Mass: ...
    def daughter_2_mass(self) -> Mass: ...
    def daughter_mass(self, daughter: str) -> Mass: ...
    def costheta(self, daughter: str, frame: _Frame = 'Helicity') -> CosTheta: ...
    def phi(self, daughter: str, frame: _Frame = 'Helicity') -> Phi: ...
    def angles(self, daughter: str, frame: _Frame = 'Helicity') -> Angles: ...
    def helicity_factor(
        self,
        *tags: str,
        spin: JLike,
        projection: MLike,
        daughter: str,
        lambda_1: MLike,
        lambda_2: MLike,
        frame: _Frame = 'Helicity',
    ) -> Expression: ...
    def canonical_factor(
        self,
        *tags: str,
        spin: JLike,
        projection: MLike,
        orbital_l: LLike,
        coupled_spin: JLike,
        daughter: str,
        daughter_1_spin: JLike,
        daughter_2_spin: JLike,
        lambda_1: MLike,
        lambda_2: MLike,
        frame: _Frame = 'Helicity',
    ) -> Expression: ...

class Production:
    reaction: Reaction
    produced: str
    recoil: str

    def costheta(self) -> CosTheta: ...
    def phi(self) -> Phi: ...
    def angles(self) -> Angles: ...
    def helicity_factor(
        self,
        *tags: str,
        spin: JLike,
        projection: MLike,
        lambda_produced: MLike,
        lambda_recoil: MLike,
    ) -> Expression: ...
    def canonical_factor(
        self,
        *tags: str,
        spin: JLike,
        projection: MLike,
        orbital_l: LLike,
        coupled_spin: JLike,
        produced_spin: JLike,
        recoil_spin: JLike,
        lambda_produced: MLike,
        lambda_recoil: MLike,
    ) -> Expression: ...

class Reaction:
    @staticmethod
    def two_to_two(
        p1: Particle, p2: Particle, p3: Particle, p4: Particle
    ) -> Reaction: ...
    def mass(self, particle: str) -> Mass: ...
    def decay(self, parent: str) -> Decay: ...
    def production(self) -> Production: ...
    def mandelstam(
        self, channel: Literal['s', 't', 'u', 'S', 'T', 'U']
    ) -> Mandelstam: ...
    def pol_angle(self, pol_angle: str) -> PolAngle: ...
    def polarization(self, *, pol_magnitude: str, pol_angle: str) -> Polarization: ...

__all__ = ['Decay', 'Particle', 'Production', 'Reaction']
