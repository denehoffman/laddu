from __future__ import annotations

from typing import Protocol, cast

from . import amplitudes, data, experimental, extensions, mpi, utils
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
    AutocorrelationTerminator,
    ControlFlow,
    EnsembleStatus,
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
    Swarm,
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


_laddu = cast('_BackendProtocol', _backend_module)

DatasetBase = Dataset

__doc__: str | None = _laddu.__doc__
__version__: str = _laddu.version()
available_parallelism = _laddu.available_parallelism


__all__ = [
    'NLL',
    'Angles',
    'AutocorrelationTerminator',
    'BinnedDataset',
    'BreitWigner',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'DatasetBase',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'Expression',
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
    'Swarm',
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
    'integrated_autocorrelation_times',
    'likelihood_product',
    'likelihood_sum',
    'mpi',
    'parameter',
    'utils',
]
