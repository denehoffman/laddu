"""Quantum-number helpers."""

from fractions import Fraction
from typing import TypeAlias

from laddu.laddu import (
    AllowedPartialWave,
    Charge,
    ExternalId,
    Isospin,
    J,
    L,
    M,
    Parity,
    PartialWave,
    ParticleProperties,
    Reflectivity,
    RuleSet,
    S,
    SelectionRules,
    Statistics,
    allowed_partial_waves,
    allowed_projections,
    coupled_spins,
)

ScalarQuantumNumber: TypeAlias = int | float | Fraction
JLike: TypeAlias = ScalarQuantumNumber | J
LLike: TypeAlias = ScalarQuantumNumber | L
MLike: TypeAlias = ScalarQuantumNumber | M
QuantumNumber: TypeAlias = ScalarQuantumNumber | J | L | M

__all__ = [
    'AllowedPartialWave',
    'Charge',
    'ExternalId',
    'Isospin',
    'J',
    'JLike',
    'L',
    'LLike',
    'M',
    'MLike',
    'Parity',
    'PartialWave',
    'ParticleProperties',
    'QuantumNumber',
    'Reflectivity',
    'RuleSet',
    'S',
    'ScalarQuantumNumber',
    'SelectionRules',
    'Statistics',
    'allowed_partial_waves',
    'allowed_projections',
    'coupled_spins',
]
