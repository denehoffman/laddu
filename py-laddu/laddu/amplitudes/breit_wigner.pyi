from typing import Literal, overload

from laddu.amplitudes import Expression, ParameterLike
from laddu.utils.variables import Mass

@overload
def BreitWigner(
    name: str,
    mass: ParameterLike,
    width: ParameterLike,
    l: Literal[0, 1, 2, 3, 4],
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    barrier_factors: bool = True,
) -> Expression: ...
@overload
def BreitWigner(
    name: str,
    mass: ParameterLike,
    width: ParameterLike,
    l: int,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    barrier_factors: bool = True,
) -> Expression: ...
def BreitWignerNonRelativistic(
    name: str,
    mass: ParameterLike,
    width: ParameterLike,
    resonance_mass: Mass,
) -> Expression: ...
