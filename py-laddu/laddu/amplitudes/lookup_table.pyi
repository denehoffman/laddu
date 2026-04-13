from collections.abc import Sequence
from typing import Literal

from laddu.amplitudes import Expression
from laddu.utils.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude

def LookupTable(
    name: str,
    variables: Sequence[Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam],
    axes: Sequence[Sequence[float]],
    values: Sequence[complex],
    interpolation: Literal['nearest', 'step', 'bin'] = 'nearest',
    boundary_mode: Literal['zero', 'zero_outside', 'zero-outside', 'clamp'] = 'zero',
) -> Expression: ...
