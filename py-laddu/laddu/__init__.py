from __future__ import annotations as _annotations

from collections.abc import Iterator as _Iterator
from contextlib import contextmanager as _contextmanager
from typing import Protocol as _Protocol
from typing import cast as _cast

from . import amplitudes, data, experimental, extensions, io, mpi, utils
from ._backend import backend as _backend_module
from .amplitudes import One, Zero, constant, expr_product, expr_sum, parameter
from .amplitudes.breit_wigner import BreitWigner
from .amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar
from .amplitudes.phase_space import PhaseSpaceFactor
from .amplitudes.ylm import Ylm
from .amplitudes.zlm import PolPhase, Zlm
from .data import BinnedDataset, Dataset, Event
from .extensions import (
    NLL,
    ControlFlow,
    EnsembleStatus,
    GradientFreeStatus,
    GradientStatus,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    MCMCObserver,
    MCMCSummary,
    MCMCTerminator,
    MinimizationObserver,
    MinimizationStatus,
    MinimizationSummary,
    MinimizationTerminator,
    StochasticNLL,
    integrated_autocorrelation_times,
    likelihood_product,
    likelihood_sum,
)
from .laddu import Evaluator, Expression, ParameterLike
from .utils.variables import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    Topology,
)
from .utils.vectors import Vec3, Vec4


class _BackendProtocol(_Protocol):
    __doc__: str | None

    def version(self) -> str: ...

    def available_parallelism(self) -> int: ...

    def get_threads(self) -> int: ...

    def set_threads(self, n_threads: int) -> None: ...


_laddu = _cast('_BackendProtocol', _backend_module)

__doc__: str | None = _laddu.__doc__
__version__: str = _laddu.version()
available_parallelism = _laddu.available_parallelism
get_threads = _laddu.get_threads
set_threads = _laddu.set_threads


@_contextmanager
def threads(n_threads: int) -> _Iterator[None]:
    """Temporarily override the global default thread count within a ``with`` block."""
    previous = get_threads()
    set_threads(n_threads)
    try:
        yield
    finally:
        set_threads(previous)


__all__ = [
    'NLL',
    'Angles',
    'BinnedDataset',
    'BreitWigner',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'Expression',
    'GradientFreeStatus',
    'GradientStatus',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'MCMCObserver',
    'MCMCSummary',
    'MCMCTerminator',
    'Mandelstam',
    'Mass',
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'One',
    'ParameterLike',
    'PhaseSpaceFactor',
    'Phi',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'Scalar',
    'StochasticNLL',
    'Topology',
    'Vec3',
    'Vec4',
    'Ylm',
    'Zero',
    'Zlm',
    '__version__',
    'amplitudes',
    'constant',
    'data',
    'experimental',
    'expr_product',
    'expr_sum',
    'extensions',
    'get_threads',
    'integrated_autocorrelation_times',
    'io',
    'likelihood_product',
    'likelihood_sum',
    'mpi',
    'parameter',
    'set_threads',
    'threads',
    'utils',
]
