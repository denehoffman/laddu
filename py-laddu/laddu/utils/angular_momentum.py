"""Angular-momentum utility functions."""

from fractions import Fraction
from typing import TypeAlias

from laddu.laddu import allowed_projections, helicity_combinations

QuantumNumber: TypeAlias = int | float | Fraction

__all__ = [
    'QuantumNumber',
    'allowed_projections',
    'helicity_combinations',
]
