from laddu.amplitudes import Expression, Parameter
from laddu.utils.variables import Mass

def Flatte(
    name: str,
    mass: Parameter,
    observed_channel_coupling: Parameter,
    alternate_channel_coupling: Parameter,
    observed_channel_daughter_masses: tuple[Mass, Mass],
    alternate_channel_daughter_masses: tuple[float, float],
    resonance_mass: Mass,
) -> Expression: ...
