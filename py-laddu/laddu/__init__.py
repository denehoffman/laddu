from __future__ import annotations as _annotations

from collections.abc import Iterator as _Iterator
from contextlib import contextmanager as _contextmanager
from typing import Protocol as _Protocol
from typing import cast as _cast

from . import (
    amplitude,
    amplitudes,
    data,
    experimental,
    extensions,
    generation,
    io,
    likelihood,
    math,
    mpi,
    optimize,
    quantum,
    reaction,
    utils,
    variables,
    vectors,
)
from ._backend import backend as _backend_module
from .amplitude import (
    CompiledExpression,
    Evaluator,
    Expression,
    One,
    Parameter,
    ParameterMap,
    TestAmplitude,
    Zero,
    expr_product,
    expr_sum,
    parameter,
)
from .amplitudes.angular import (
    BlattWeisskopf,
    ClebschGordan,
    PhotonSDME,
    PolPhase,
    Wigner3j,
    WignerD,
    Ylm,
    Zlm,
)
from .amplitudes.kmatrix import (
    KopfKMatrixA0,
    KopfKMatrixA0Channel,
    KopfKMatrixA2,
    KopfKMatrixA2Channel,
    KopfKMatrixF0,
    KopfKMatrixF0Channel,
    KopfKMatrixF2,
    KopfKMatrixF2Channel,
    KopfKMatrixPi1,
    KopfKMatrixPi1Channel,
    KopfKMatrixRho,
    KopfKMatrixRhoChannel,
)
from .amplitudes.resonance import (
    BreitWigner,
    BreitWignerNonRelativistic,
    Flatte,
    PhaseSpaceFactor,
    Voigt,
)
from .amplitudes.scalar import ComplexScalar, PolarComplexScalar, Scalar, VariableScalar
from .data import BinnedDataset, Dataset, Event
from .generation import (
    CompositeGenerator,
    Distribution,
    EventGenerator,
    GeneratedBatch,
    GeneratedBatchIter,
    GeneratedEventLayout,
    GeneratedParticle,
    GeneratedParticleLayout,
    GeneratedReaction,
    GeneratedStorage,
    GeneratedVertexLayout,
    InitialGenerator,
    MandelstamTDistribution,
    ParticleSpecies,
    Reconstruction,
    StableGenerator,
)
from .likelihood import (
    NLL,
    LikelihoodExpression,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodZero,
    StochasticNLL,
    likelihood_product,
    likelihood_sum,
)
from .math import Histogram
from .optimize import (
    ControlFlow,
    EnsembleStatus,
    GradientFreeStatus,
    GradientStatus,
    MCMCObserver,
    MCMCSummary,
    MCMCTerminator,
    MinimizationObserver,
    MinimizationStatus,
    MinimizationSummary,
    MinimizationTerminator,
    integrated_autocorrelation_times,
)
from .quantum import (
    AllowedPartialWave,
    Charge,
    Isospin,
    Parity,
    PartialWave,
    ParticleProperties,
    RuleSet,
    SelectionRules,
    Statistics,
    allowed_partial_waves,
    allowed_projections,
    coupled_spins,
)
from .reaction import Decay, Particle, Production, Reaction
from .variables import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
)
from .vectors import Vec3, Vec4


class _BackendProtocol(_Protocol):
    __doc__: str | None

    def version(self) -> str: ...

    def available_parallelism(self) -> int: ...

    def get_threads(self) -> int: ...

    def set_threads(self, n_threads: int | None) -> None: ...


_laddu = _cast('_BackendProtocol', _backend_module)

__doc__: str | None = _laddu.__doc__
__version__: str = _laddu.version()
available_parallelism = _laddu.available_parallelism
get_threads = _laddu.get_threads
set_threads = _laddu.set_threads


@_contextmanager
def threads(n_threads: int | None) -> _Iterator[None]:
    """Temporarily override the global default thread count within a ``with`` block."""
    previous = get_threads()
    set_threads(n_threads)
    try:
        yield
    finally:
        set_threads(previous)


__all__ = [
    'NLL',
    'AllowedPartialWave',
    'Angles',
    'BinnedDataset',
    'BlattWeisskopf',
    'BreitWigner',
    'BreitWignerNonRelativistic',
    'Charge',
    'ClebschGordan',
    'CompiledExpression',
    'ComplexScalar',
    'CompositeGenerator',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'Decay',
    'Distribution',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'EventGenerator',
    'Expression',
    'Flatte',
    'GeneratedBatch',
    'GeneratedBatchIter',
    'GeneratedEventLayout',
    'GeneratedParticle',
    'GeneratedParticleLayout',
    'GeneratedReaction',
    'GeneratedStorage',
    'GeneratedVertexLayout',
    'GradientFreeStatus',
    'GradientStatus',
    'Histogram',
    'InitialGenerator',
    'Isospin',
    'KopfKMatrixA0',
    'KopfKMatrixA0Channel',
    'KopfKMatrixA2',
    'KopfKMatrixA2Channel',
    'KopfKMatrixF0',
    'KopfKMatrixF0Channel',
    'KopfKMatrixF2',
    'KopfKMatrixF2Channel',
    'KopfKMatrixPi1',
    'KopfKMatrixPi1Channel',
    'KopfKMatrixRho',
    'KopfKMatrixRhoChannel',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'MCMCObserver',
    'MCMCSummary',
    'MCMCTerminator',
    'Mandelstam',
    'MandelstamTDistribution',
    'Mass',
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'One',
    'Parameter',
    'ParameterMap',
    'Parity',
    'PartialWave',
    'Particle',
    'ParticleProperties',
    'ParticleSpecies',
    'PhaseSpaceFactor',
    'Phi',
    'PhotonSDME',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'Production',
    'Reaction',
    'Reconstruction',
    'RuleSet',
    'Scalar',
    'SelectionRules',
    'StableGenerator',
    'Statistics',
    'StochasticNLL',
    'TestAmplitude',
    'VariableScalar',
    'Vec3',
    'Vec4',
    'Voigt',
    'Wigner3j',
    'WignerD',
    'Ylm',
    'Zero',
    'Zlm',
    '__version__',
    'allowed_partial_waves',
    'allowed_projections',
    'amplitude',
    'amplitudes',
    'coupled_spins',
    'data',
    'experimental',
    'expr_product',
    'expr_sum',
    'extensions',
    'generation',
    'get_threads',
    'integrated_autocorrelation_times',
    'io',
    'likelihood',
    'likelihood_product',
    'likelihood_sum',
    'math',
    'mpi',
    'optimize',
    'parameter',
    'quantum',
    'reaction',
    'set_threads',
    'threads',
    'utils',
    'variables',
    'vectors',
]
