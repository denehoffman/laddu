from laddu.amplitudes import Expression, Parameter
from laddu.utils.variables import Mass

def Voigt(
    name: str,
    mass: Parameter,
    width: Parameter,
    sigma: Parameter,
    resonance_mass: Mass,
) -> Expression: ...
