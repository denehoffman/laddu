from laddu.amplitudes import Amplitude
from laddu.utils.variables import Mandelstam, Mass

def PhaseSpaceFactor(
    name: str,
    recoil_mass: Mass,
    daughter_1_mass: Mass,
    daughter_2_mass: Mass,
    resonance_mass: Mass,
    mandelstam_s: Mandelstam,
) -> Amplitude: ...
