from fractions import Fraction
from typing import TypeAlias

QuantumNumber: TypeAlias = int | float | Fraction

def allowed_projections(spin: QuantumNumber) -> list[int | Fraction]: ...
def helicity_combinations(
    spin_1: QuantumNumber, spin_2: QuantumNumber
) -> list[tuple[int | Fraction, int | Fraction, int | Fraction]]: ...

__all__ = [
    'QuantumNumber',
    'allowed_projections',
    'helicity_combinations',
]
