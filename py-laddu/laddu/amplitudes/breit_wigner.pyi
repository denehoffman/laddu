from typing import Literal, overload

from laddu.amplitudes import Amplitude, ParameterLike
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
) -> Amplitude: ...
@overload
def BreitWigner(
    name: str,
    mass: ParameterLike,
    width: ParameterLike,
    l: int,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
) -> Amplitude: ...
