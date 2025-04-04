from typing import Literal

import numpy.typing as npt

from laddu.extensions import NLL, LikelihoodTerm
from laddu.utils.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude

def BinnedGuideTerm(
    nll: NLL,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    amplitude_sets: list[list[str]],
    bins: int,
    range: tuple[float, float],
    count_sets: list[list[float]] | list[npt.NDArray],
    error_sets: list[list[float]] | list[npt.NDArray] | None,
) -> LikelihoodTerm: ...
def Regularizer(
    parameters: list[str],
    lda: float,
    p: Literal[1, 2],
    weights: npt.NDArray | None = None,
) -> LikelihoodTerm: ...

__all__ = ['BinnedGuideTerm', 'Regularizer']
