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
    gen,
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
    InitialValue,
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
    PhotonSDME,
    PolPhase,
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
    DatasetSink,
    DecayParticlePlan,
    DecayPlan,
    EventGenerator,
    GeneratedEvent,
    GenerationOptions,
    GenerationOutput,
    GenerationPlan,
    GenerationResult,
    GenerationStats,
    InitialParticlePlan,
    MassSampler,
    MomentumSource,
    PlannedMass,
    ProductionPlan,
    Raw,
    VertexGenerator,
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
from .math import Histogram, clebsch_gordan
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
    J,
    L,
    M,
    Parity,
    PartialWave,
    ParticleProperties,
    Reflectivity,
    RuleSet,
    S,
    SelectionRules,
    Statistics,
    allowed_partial_waves,
    allowed_projections,
    coupled_spins,
)
from .reaction import (
    Axes,
    Axis,
    Channel,
    Frame,
    Particle,
    ParticleSource,
    TwoBodyCoupling,
    Vertex,
)
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
    'Axes',
    'Axis',
    'BinnedDataset',
    'BlattWeisskopf',
    'BreitWigner',
    'BreitWignerNonRelativistic',
    'Channel',
    'Charge',
    'CompiledExpression',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'DatasetSink',
    'DecayParticlePlan',
    'DecayPlan',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'EventGenerator',
    'Expression',
    'Flatte',
    'Frame',
    'GeneratedEvent',
    'GenerationOptions',
    'GenerationOutput',
    'GenerationPlan',
    'GenerationResult',
    'GenerationStats',
    'GradientFreeStatus',
    'GradientStatus',
    'Histogram',
    'InitialParticlePlan',
    'InitialValue',
    'Isospin',
    'J',
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
    'L',
    'LikelihoodExpression',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodZero',
    'M',
    'MCMCObserver',
    'MCMCSummary',
    'MCMCTerminator',
    'Mandelstam',
    'Mass',
    'MassSampler',
    'MinimizationObserver',
    'MinimizationStatus',
    'MinimizationSummary',
    'MinimizationTerminator',
    'MomentumSource',
    'One',
    'Parameter',
    'ParameterMap',
    'Parity',
    'PartialWave',
    'Particle',
    'ParticleProperties',
    'ParticleSource',
    'PhaseSpaceFactor',
    'Phi',
    'PhotonSDME',
    'PlannedMass',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'ProductionPlan',
    'Raw',
    'Reflectivity',
    'RuleSet',
    'S',
    'Scalar',
    'SelectionRules',
    'Statistics',
    'StochasticNLL',
    'TestAmplitude',
    'TwoBodyCoupling',
    'VariableScalar',
    'Vec3',
    'Vec4',
    'Vertex',
    'VertexGenerator',
    'Voigt',
    'WignerD',
    'Ylm',
    'Zero',
    'Zlm',
    '__version__',
    'allowed_partial_waves',
    'allowed_projections',
    'amplitude',
    'amplitudes',
    'clebsch_gordan',
    'coupled_spins',
    'data',
    'experimental',
    'expr_product',
    'expr_sum',
    'extensions',
    'gen',
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
