from fractions import Fraction
from typing import TypeAlias

from laddu.amplitude import Expression
from laddu.reaction import Decay
from laddu.variables import Angles, Polarization

QuantumNumber: TypeAlias = int | float | Fraction

def WignerD(
    name: str,
    spin: QuantumNumber,
    row_projection: QuantumNumber,
    column_projection: QuantumNumber,
    angles: Angles,
) -> Expression: ...
def BlattWeisskopf(
    name: str,
    decay: Decay,
    l: QuantumNumber,
    reference_mass: float,
    q_r: float = ...,
    sheet: str = ...,
    kind: str = ...,
) -> Expression: ...
def ClebschGordan(
    name: str,
    j1: QuantumNumber,
    m1: QuantumNumber,
    j2: QuantumNumber,
    m2: QuantumNumber,
    j: QuantumNumber,
    m: QuantumNumber,
) -> Expression: ...
def Wigner3j(
    name: str,
    j1: QuantumNumber,
    m1: QuantumNumber,
    j2: QuantumNumber,
    m2: QuantumNumber,
    j3: QuantumNumber,
    m3: QuantumNumber,
) -> Expression: ...
def PhotonSDME(
    name: str,
    helicity: int,
    helicity_prime: int,
    polarization: Polarization | None = ...,
) -> Expression: ...
def Ylm(name: str, l: int, m: int, angles: Angles) -> Expression: ...
def Zlm(
    name: str,
    l: int,
    m: int,
    r: str,
    angles: Angles,
    polarization: Polarization,
) -> Expression: ...
def PolPhase(
    name: str,
    polarization: Polarization,
) -> Expression: ...

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
