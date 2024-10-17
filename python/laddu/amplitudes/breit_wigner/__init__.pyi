from typing import Literal

from laddu.amplitudes import Amplitude, ParameterLike
from laddu.utils.variables import Mass

def BreitWigner(  # noqa: N802
    name: str,
    mass: ParameterLike,
    width: ParameterLike,
    l: Literal[0, 1, 2, 3, 4],  # noqa: E741
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
) -> Amplitude: ...
