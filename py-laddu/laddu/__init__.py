from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

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
    AdamSettings,
    AIESMoveConfig,
    AIESSettings,
    AutocorrelationTerminator,
    ControlFlow,
    EnsembleStatus,
    ESSMoveConfig,
    ESSSettings,
    LBFGSBSettings,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    LineSearchConfig,
    MCMCObserver,
    MCMCSummary,
    MCMCTerminator,
    MinimizationObserver,
    MinimizationStatus,
    MinimizationSummary,
    MinimizationTerminator,
    NelderMeadSettings,
    PSOSettings,
    SimplexConfig,
    StochasticNLL,
    Swarm,
    SwarmInitializerConfig,
    SwarmParticle,
    Walker,
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


class _BackendProtocol(Protocol):
    __doc__: str | None

    def version(self) -> str: ...

    def available_parallelism(self) -> int: ...

    def get_threads(self) -> int: ...

    def set_threads(self, n_threads: int) -> None: ...


_laddu = cast('_BackendProtocol', _backend_module)

__doc__: str | None = _laddu.__doc__
__version__: str = _laddu.version()
available_parallelism = _laddu.available_parallelism
get_threads = _laddu.get_threads
set_threads = _laddu.set_threads


@contextmanager
def threads(n_threads: int) -> Iterator[None]:
    """Temporarily override the global default thread count within a ``with`` block."""
    previous = get_threads()
    set_threads(n_threads)
    try:
        yield
    finally:
        set_threads(previous)


__all__ = [
    'NLL',
    'AIESMoveConfig',
    'AIESSettings',
    'AdamSettings',
    'Angles',
    'AutocorrelationTerminator',
    'BinnedDataset',
    'BreitWigner',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'ESSMoveConfig',
    'ESSSettings',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'Expression',
    'LBFGSBSettings',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'LineSearchConfig',
    'MCMCObserver',
    'MCMCSummary',
    'MCMCTerminator',
    'Mandelstam',
    'Mass',
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'NelderMeadSettings',
    'One',
    'PSOSettings',
    'ParameterLike',
    'PhaseSpaceFactor',
    'Phi',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'Scalar',
    'SimplexConfig',
    'StochasticNLL',
    'Swarm',
    'SwarmInitializerConfig',
    'SwarmParticle',
    'Topology',
    'Vec3',
    'Vec4',
    'Walker',
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
