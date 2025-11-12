
from pathlib import Path

from laddu.amplitudes import (
    Amplitude,
    AmplitudeID,
    AmplitudeOne,
    AmplitudeZero,
    Evaluator,
    Expression,
    Manager,
    Model,
    ParameterLike,
    TestAmplitude,
    amplitude_product,
    amplitude_sum,
    constant,
    parameter,
)
from laddu.amplitudes.breit_wigner import BreitWigner
from laddu.amplitudes.common import ComplexScalar, PolarComplexScalar, Scalar
from laddu.amplitudes.kmatrix import (
    KopfKMatrixA0,
    KopfKMatrixA2,
    KopfKMatrixF0,
    KopfKMatrixF2,
    KopfKMatrixPi1,
    KopfKMatrixRho,
)
from laddu.amplitudes.phase_space import PhaseSpaceFactor
from laddu.amplitudes.piecewise import (
    PiecewiseComplexScalar,
    PiecewisePolarComplexScalar,
    PiecewiseScalar,
)
from laddu.amplitudes.ylm import Ylm
from laddu.amplitudes.zlm import PolPhase, Zlm
from laddu.data import BinnedDataset, Event
from laddu.experimental import BinnedGuideTerm, Regularizer
from laddu.extensions import (
    NLL,
    AutocorrelationTerminator,
    ControlFlow,
    EnsembleStatus,
    LikelihoodEvaluator,
    LikelihoodExpression,
    LikelihoodID,
    LikelihoodManager,
    LikelihoodOne,
    LikelihoodScalar,
    LikelihoodTerm,
    LikelihoodZero,
    MCMCSummary,
    MinimizationStatus,
    MinimizationSummary,
    StochasticNLL,
    Swarm,
    SwarmParticle,
    Walker,
    integrated_autocorrelation_times,
    likelihood_product,
    likelihood_sum,
)
from laddu.utils.variables import (
    Angles,
    CosTheta,
    Mandelstam,
    Mass,
    Phi,
    PolAngle,
    Polarization,
    PolMagnitude,
    VariableExpression,
)
from laddu.utils.vectors import Vec3, Vec4

class Dataset:
    events: list[Event]
    p4_names: list[str]
    aux_names: list[str]
    n_events: int
    n_events_weighted: float

    def __init__(
        self,
        events: list[Event],
        *,
        p4_names: list[str] | None = None,
        aux_names: list[str] | None = None,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Event: ...
    def boost_to_rest_frame_of(self, names: list[str]) -> Dataset: ...
    @staticmethod
    def open(
        path: str | Path,
        *,
        p4s: list[str] | None = None,
        aux: list[str] | None = None,
        boost_to_restframe_of: list[str] | None = None,
        tree: str | None = None,
    ) -> Dataset: ...

DatasetBase = Dataset

def version() -> str: ...
def available_parallelism() -> int: ...
def use_mpi(*, trigger: bool) -> None: ...
def using_mpi() -> bool: ...
def finalize_mpi() -> None: ...
def is_root() -> bool: ...
def get_rank() -> int: ...
def get_size() -> int: ...
def is_mpi_available() -> bool: ...

__all__ = [
    'NLL',
    'Amplitude',
    'AmplitudeID',
    'AmplitudeOne',
    'AmplitudeZero',
    'Angles',
    'AutocorrelationTerminator',
    'BinnedDataset',
    'BinnedGuideTerm',
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
    'KopfKMatrixA0',
    'KopfKMatrixA2',
    'KopfKMatrixF0',
    'KopfKMatrixF2',
    'KopfKMatrixPi1',
    'KopfKMatrixRho',
    'LikelihoodEvaluator',
    'LikelihoodExpression',
    'LikelihoodID',
    'LikelihoodManager',
    'LikelihoodOne',
    'LikelihoodScalar',
    'LikelihoodTerm',
    'LikelihoodZero',
    'MCMCSummary',
    'Manager',
    'Mandelstam',
    'Mass',
    'MinimizationStatus',
    'MinimizationSummary',
    'Model',
    'ParameterLike',
    'PhaseSpaceFactor',
    'Phi',
    'PiecewiseComplexScalar',
    'PiecewisePolarComplexScalar',
    'PiecewiseScalar',
    'PolAngle',
    'PolMagnitude',
    'PolPhase',
    'PolarComplexScalar',
    'Polarization',
    'Regularizer',
    'Scalar',
    'StochasticNLL',
    'Swarm',
    'SwarmParticle',
    'TestAmplitude',
    'VariableExpression',
    'Vec3',
    'Vec4',
    'Walker',
    'Ylm',
    'Zlm',
    'amplitude_product',
    'amplitude_sum',
    'available_parallelism',
    'constant',
    'finalize_mpi',
    'get_rank',
    'get_size',
    'integrated_autocorrelation_times',
    'is_mpi_available',
    'is_root',
    'likelihood_product',
    'likelihood_sum',
    'parameter',
    'use_mpi',
    'using_mpi',
    'version',
]
