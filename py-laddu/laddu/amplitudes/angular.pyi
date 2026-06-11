from laddu.amplitude import Expression
from laddu.quantum import JLike, LLike, MLike, QuantumNumber, Reflectivity, Sign
from laddu.variables import Angles, Mass, Polarization

def WignerD(
    *tags: str,
    spin: JLike,
    row_projection: MLike,
    column_projection: MLike,
    angles: Angles,
) -> Expression: ...
def BlattWeisskopf(
    *tags: str,
    parent_mass: Mass,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    l: LLike,
    reference_mass: float,
    q_r: float = ...,
    sheet: str = ...,
    kind: str = ...,
) -> Expression: ...
def PhotonSDME(
    *tags: str,
    helicity: MLike,
    helicity_prime: MLike,
    polarization: Polarization | None = ...,
) -> Expression: ...
def Ylm(*tags: str, l: LLike, m: MLike, angles: Angles) -> Expression: ...
def Zlm(
    *tags: str,
    l: LLike,
    m: MLike,
    r: Reflectivity | Sign,
    angles: Angles,
    polarization: Polarization,
) -> Expression: ...
def PolPhase(
    *tags: str,
    polarization: Polarization,
) -> Expression: ...

__all__ = [
    'BlattWeisskopf',
    'PhotonSDME',
    'PolPhase',
    'QuantumNumber',
    'WignerD',
    'Ylm',
    'Zlm',
]
