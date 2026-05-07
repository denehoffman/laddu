"""Angular amplitudes and related factors."""

from fractions import Fraction
from typing import TypeAlias

from laddu.laddu import (
    BlattWeisskopf,
    ClebschGordan,
    PhotonSDME,
    PolPhase,
    Wigner3j,
    WignerD,
    Ylm,
    Zlm,
)

QuantumNumber: TypeAlias = int | float | Fraction

__all__ = [
    'BlattWeisskopf',
    'ClebschGordan',
    'PhotonSDME',
    'PolPhase',
    'QuantumNumber',
    'Wigner3j',
    'WignerD',
    'Ylm',
    'Zlm',
]
