"""Quantum-number helpers."""

from fractions import Fraction
from typing import TypeAlias

from laddu.laddu import (
    AllowedPartialWave,
    Charge,
    Isospin,
    Parity,
    PartialWave,
    ParticleProperties,
    RuleSet,
    SelectionRules,
    Statistics,
    allowed_partial_waves,
    allowed_projections,
    coupled_spins,
)

QuantumNumber: TypeAlias = int | float | Fraction

__all__ = [
    'AllowedPartialWave',
    'Charge',
    'Isospin',
    'Parity',
    'PartialWave',
    'ParticleProperties',
    'QuantumNumber',
    'RuleSet',
    'SelectionRules',
    'Statistics',
    'allowed_partial_waves',
    'allowed_projections',
    'coupled_spins',
]
