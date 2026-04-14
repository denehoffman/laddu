from contextlib import AbstractContextManager as AbstractContextManager

from . import amplitudes, data, experimental, extensions, io, mpi, utils
from .amplitudes import (
    Evaluator,
    Expression,
    One,
    ParameterLike,
    Zero,
    constant,
    expr_product,
    expr_sum,
    parameter,
)
from .amplitudes.breit_wigner import BreitWigner, BreitWignerNonRelativistic
from .amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar, VariableScalar
from .amplitudes.flatte import Flatte
from .amplitudes.phase_space import PhaseSpaceFactor
from .amplitudes.voigt import Voigt
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

__version__: str

def available_parallelism() -> int: ...
def get_threads() -> int: ...
def set_threads(n_threads: int) -> None: ...
def threads(n_threads: int) -> AbstractContextManager[None, None]: ...

__all__ = [
    'NLL',
    'Angles',
    'BinnedDataset',
    'BreitWigner',
    'BreitWignerNonRelativistic',
    'ComplexScalar',
    'ControlFlow',
    'CosTheta',
    'Dataset',
    'EnsembleStatus',
    'Evaluator',
    'Event',
    'Expression',
    'Flatte',
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
    'VariableScalar',
    'Vec3',
    'Vec4',
    'Voigt',
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
