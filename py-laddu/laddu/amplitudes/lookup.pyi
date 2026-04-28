from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from laddu.amplitude import Expression, Parameter
from laddu.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude

def LookupTable(
    name: str,
    variables: Sequence[Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam],
    axis_coordinates: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    values: Sequence[complex] | npt.NDArray[np.complex128],
    interpolation: Literal['nearest', 'step', 'bin', 'linear', 'multilinear'] = 'nearest',
    boundary_mode: Literal['zero', 'zero_outside', 'zero-outside', 'clamp'] = 'zero',
) -> Expression: ...
def LookupTableScalar(
    name: str,
    variables: Sequence[Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam],
    axis_coordinates: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    values: Sequence[Parameter],
    interpolation: Literal['nearest', 'step', 'bin', 'linear', 'multilinear'] = 'nearest',
    boundary_mode: Literal['zero', 'zero_outside', 'zero-outside', 'clamp'] = 'zero',
) -> Expression: ...
def LookupTableComplex(
    name: str,
    variables: Sequence[Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam],
    axis_coordinates: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    values: Sequence[tuple[Parameter, Parameter]],
    interpolation: Literal['nearest', 'step', 'bin', 'linear', 'multilinear'] = 'nearest',
    boundary_mode: Literal['zero', 'zero_outside', 'zero-outside', 'clamp'] = 'zero',
) -> Expression: ...
def LookupTablePolar(
    name: str,
    variables: Sequence[Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam],
    axis_coordinates: Sequence[Sequence[float]] | npt.NDArray[np.float64],
    values: Sequence[tuple[Parameter, Parameter]],
    interpolation: Literal['nearest', 'step', 'bin', 'linear', 'multilinear'] = 'nearest',
    boundary_mode: Literal['zero', 'zero_outside', 'zero-outside', 'clamp'] = 'zero',
) -> Expression: ...
