from laddu.amplitudes import Expression, ParameterLike
from laddu.utils.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude

def PiecewiseScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    bins: int,
    range: tuple[float, float],
    values: list[ParameterLike],
) -> Expression: ...
def PiecewiseComplexScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    bins: int,
    range: tuple[float, float],
    values: list[tuple[ParameterLike, ParameterLike]],
) -> Expression: ...
def PiecewisePolarComplexScalar(
    name: str,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    bins: int,
    range: tuple[float, float],
    values: list[tuple[ParameterLike, ParameterLike]],
) -> Expression: ...
