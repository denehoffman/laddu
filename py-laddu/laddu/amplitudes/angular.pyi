from laddu.amplitude import Expression
from laddu.quantum import JLike, LLike, MLike, QuantumNumber
from laddu.reaction import Decay
from laddu.variables import Angles, Polarization

def WignerD(
    *tags: str,
    spin: JLike,
    row_projection: MLike,
    column_projection: MLike,
    angles: Angles,
) -> Expression: ...
def BlattWeisskopf(
    *tags: str,
    decay: Decay,
    l: LLike,
    reference_mass: float,
    q_r: float = ...,
    sheet: str = ...,
    kind: str = ...,
) -> Expression: ...
def ClebschGordan(
    *tags: str,
    j1: JLike,
    m1: MLike,
    j2: JLike,
    m2: MLike,
    j: JLike,
    m: MLike,
) -> Expression: ...
def Wigner3j(
    *tags: str,
    j1: JLike,
    m1: MLike,
    j2: JLike,
    m2: MLike,
    j3: JLike,
    m3: MLike,
) -> Expression: ...
def PhotonSDME(
    *tags: str,
    helicity: int,
    helicity_prime: int,
    polarization: Polarization | None = ...,
) -> Expression: ...
def Ylm(*tags: str, l: int, m: int, angles: Angles) -> Expression: ...
def Zlm(
    *tags: str,
    l: int,
    m: int,
    r: str,
    angles: Angles,
    polarization: Polarization,
) -> Expression: ...
def PolPhase(
    *tags: str,
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
