from laddu.amplitudes import Expression, ParameterLike
from laddu.utils.variables import Mass

def Flatte(
    name: str,
    mass: ParameterLike,
    observed_channel_coupling: ParameterLike,
    alternate_channel_coupling: ParameterLike,
    observed_channel_daughter_masses: tuple[Mass, Mass],
    alternate_channel_daughter_masses: tuple[float, float],
    resonance_mass: Mass,
) -> Expression: ...
