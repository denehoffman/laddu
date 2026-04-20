from typing import Literal, overload

from laddu.amplitudes import Expression, Parameter
from laddu.utils.variables import Mass

@overload
def BreitWigner(
    name: str,
    mass: Parameter,
    width: Parameter,
    l: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8],
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    barrier_factors: bool = True,
) -> Expression: ...
@overload
def BreitWigner(
    name: str,
    mass: Parameter,
    width: Parameter,
    l: int,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    barrier_factors: bool = True,
) -> Expression: ...
def BreitWignerNonRelativistic(
    name: str,
    mass: Parameter,
    width: Parameter,
    resonance_mass: Mass,
) -> Expression: ...
