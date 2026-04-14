from laddu.amplitudes import Expression, ParameterLike
from laddu.utils.variables import Mass

def Voigt(
    name: str,
    mass: ParameterLike,
    width: ParameterLike,
    sigma: ParameterLike,
    resonance_mass: Mass,
) -> Expression: ...
