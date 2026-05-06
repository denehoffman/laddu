from laddu.amplitude import Expression, Parameter
from laddu.variables import Mandelstam, Mass

def BreitWigner(
    *tags: str,
    mass: Parameter,
    width: Parameter,
    l: int,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    barrier_factors: bool = True,
) -> Expression: ...
def BreitWignerNonRelativistic(
    *tags: str,
    mass: Parameter,
    width: Parameter,
    resonance_mass: Mass,
) -> Expression: ...
def Flatte(
    *tags: str,
    mass: Parameter,
    observed_channel_coupling: Parameter,
    alternate_channel_coupling: Parameter,
    observed_channel_daughter_masses: tuple[Mass, Mass],
    alternate_channel_daughter_masses: tuple[float, float],
    resonance_mass: Mass,
) -> Expression: ...
def Voigt(
    *tags: str,
    mass: Parameter,
    width: Parameter,
    sigma: Parameter,
    resonance_mass: Mass,
) -> Expression: ...
def PhaseSpaceFactor(
    *tags: str,
    recoil_mass: Mass,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    mandelstam_s: Mandelstam,
) -> Expression: ...
