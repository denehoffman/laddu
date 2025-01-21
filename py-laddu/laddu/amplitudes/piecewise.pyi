from laddu.amplitudes import Amplitude, ParameterLike
from laddu.utils.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude

def PiecewiseScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    bins: int,
    range: tuple[float, float],
    values: list[ParameterLike],
) -> Amplitude: ...
def PiecewiseComplexScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    bins: int,
    range: tuple[float, float],
    values: list[tuple[ParameterLike, ParameterLike]],
) -> Amplitude: ...
def PiecewisePolarComplexScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    bins: int,
    range: tuple[float, float],
    values: list[tuple[ParameterLike, ParameterLike]],
) -> Amplitude: ...
