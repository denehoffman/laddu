from collections.abc import Sequence

import numpy.typing as npt

from laddu.extensions import NLL, LikelihoodExpression
from laddu.utils.variables import CosTheta, Mandelstam, Mass, Phi, PolAngle, PolMagnitude

def BinnedGuideTerm(
    nll: NLL,
    variable: Mass | CosTheta | Phi | PolAngle | PolMagnitude | Mandelstam,
    amplitude_sets: Sequence[Sequence[str]],
    bins: int,
    range: tuple[float, float],
    count_sets: Sequence[Sequence[float]] | Sequence[npt.NDArray],
    error_sets: Sequence[Sequence[float]] | Sequence[npt.NDArray] | None,
) -> LikelihoodExpression: ...
def Regularizer(
    parameters: Sequence[str],
    lda: float,
    p: int = 1,
    weights: Sequence[float] | npt.NDArray | None = None,
) -> LikelihoodExpression: ...

__all__ = ['BinnedGuideTerm', 'Regularizer']
